"""Main module for the XRSS application."""

import asyncio
import json
import os
import random
from datetime import datetime
from typing import Any, Coroutine, Dict, List, Optional
from zoneinfo import ZoneInfo

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Response
from feedgen.feed import FeedGenerator
from redis import asyncio as aioredis
from twikit import Client as TwikitClient
from twikit import Tweet as TwikitTweet
from twikit import UserNotFound

try:
    from config import Settings
    from utils import clean_cookies, clean_tweet, setup_logging
except ImportError:
    from .config import Settings
    from .utils import clean_cookies, clean_tweet, setup_logging


# Load environment variables
load_dotenv()

# Initialize settings
settings = Settings(
    twitter_username=os.getenv("TWITTER_USERNAME"),
    twitter_email=os.getenv("TWITTER_EMAIL"),
    twitter_password=os.getenv("TWITTER_PASSWORD"),
    twitter_totp_secret=os.getenv("TWITTER_TOTP_SECRET"),
)

# Setup logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="XRSS",
    description="Convert Twitter/X feeds to RSS with custom filters and caching",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Initialize clients
twikit_client = TwikitClient("en-US")
redis = aioredis.from_url(settings.redis_url, decode_responses=True)

# Rate limiting configuration
api_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)


async def rate_limited_request(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Execute a coroutine with rate limiting."""
    async with api_semaphore:
        result = await coroutine
        await asyncio.sleep(settings.request_delay)
        return result


async def get_cached_user(username: str) -> Optional[Dict[str, Any]]:
    """Get user data from cache."""
    cache_key = f"user:{username}"
    cached_data = await redis.get(cache_key)
    return json.loads(cached_data) if cached_data else None


async def get_cached_tweets(username: str) -> List[Dict[str, Any]]:
    """Get tweets from cache or fetch if not available."""
    cache_key = f"tweets:{username}"
    cached_data = await redis.get(cache_key)

    if cached_data:
        return list(json.loads(cached_data))

    # If not in cache, fetch and cache
    await refresh_user_tweets_cache(username)
    cached_data = await redis.get(cache_key)
    return list(json.loads(cached_data)) if cached_data else []


async def refresh_user_tweets_cache(username: str) -> None:
    """Background task to refresh the cache for a user's tweets."""
    try:
        user = await rate_limited_request(twikit_client.get_user_by_screen_name(username))

        # Store user profile data
        await redis.setex(
            f"user:{username}",
            settings.cache_ttl,
            json.dumps(
                {
                    "profile_image_url": user.profile_image_url,
                    "name": user.name,
                    "screen_name": user.screen_name,
                }
            ),
        )

        # Fetch tweets with rate limiting
        tasks = [
            rate_limited_request(user.get_tweets(tweet_type="Tweets")),
            rate_limited_request(user.get_tweets(tweet_type="Replies")),
        ]
        results = await asyncio.gather(*tasks)

        # Combine and process tweets
        all_tweets = []
        seen_tweet_ids = set()
        for tweet_list in results:
            for tweet in tweet_list:
                if tweet.id not in seen_tweet_ids:
                    all_tweets.append(tweet)
                    seen_tweet_ids.add(tweet.id)

        # Sort and process tweets
        all_tweets.sort(
            key=lambda x: datetime.strptime(x.created_at, "%a %b %d %H:%M:%S +0000 %Y"),
            reverse=True,
        )

        # By default retweet's full_text is actually not full lol
        # First pass: mark tweet types and collect retweet IDs
        retweet_tasks = []
        for tweet in all_tweets:
            setattr(tweet, "type", _get_tweet_type(tweet))
            if tweet.type == "Retweet":
                retweet_tasks.append(
                    rate_limited_request(twikit_client.get_tweet_by_id(tweet.retweeted_tweet.id))
                )

        # Fetch all retweets in parallel
        retweet_map = {}
        if retweet_tasks:
            retweet_results = await asyncio.gather(*retweet_tasks)
            retweet_map = {tweet.id: clean_tweet(tweet.full_text) for tweet in retweet_results}

        # Helper function to get correct full_text for tweets
        def get_tweet_text(tweet):
            """Get the appropriate text content for a tweet, handling retweets properly."""
            if tweet.type == "Retweet" and tweet.retweeted_tweet.id in retweet_map:
                return retweet_map[tweet.retweeted_tweet.id]
            return clean_tweet(tweet.full_text)

        processed_tweets = [
            {
                "created_at": tweet.created_at,
                "type": tweet.type,
                "id": tweet.id,
                "link": f"https://x.com/{username}/status/{tweet.id}",
                "full_text": get_tweet_text(tweet),
                "in_reply_to": [
                    {
                        "id": _reply.id,
                        "full_text": clean_tweet(_reply.full_text),
                        "username": _reply.user.screen_name,
                        "user_id": _reply.user.id,
                        "created_at": _reply.created_at,
                    }
                    for _reply in (tweet.replies or [])
                ],
            }
            for tweet in all_tweets
        ]

        logger.info(f"Processed {len(processed_tweets)} tweets for {username}")

        # Store in Redis with TTL
        await redis.setex(f"tweets:{username}", settings.cache_ttl, json.dumps(processed_tweets))

    except UserNotFound:
        raise HTTPException(status_code=404, detail=f"User {username} not found")

    except Exception as e:
        logger.error(f"Error refreshing cache for {username}: {str(e)}")
        raise


def _get_tweet_type(tweet: TwikitTweet) -> str:
    """
    Determine the type of a tweet based on its characteristics.

    Args:
        tweet: The tweet object to analyze

    Returns:
        The categorized type of the tweet
    """
    if tweet.thread is not None:
        return "Thread"

    if tweet.retweeted_tweet is not None:
        return "Retweet"

    if (tweet.replies is not None and len(tweet.replies) > 0) or tweet.in_reply_to is not None:
        return "Reply"

    if tweet.is_quote_status is True:
        return "Quote"

    return "Post"


@app.post("/")
async def get_tweets(
    background_tasks: BackgroundTasks,
    usernames: List[str],
    include_posts: bool = True,
    include_replies: bool = True,
    include_retweets: bool = True,
    include_quotes: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch tweets for specified users with filtering options.

    Args:
        background_tasks: FastAPI background tasks
        usernames: List of Twitter usernames
        include_posts: Whether to include regular posts
        include_replies: Whether to include replies
        include_retweets: Whether to include retweets
        include_quotes: Whether to include quote tweets

    Returns:
        Dictionary mapping usernames to their tweets
    """

    if len(usernames) == 0:
        return {}

    # Shuffle usernames to prevent always hitting the same users first
    shuffled_usernames = usernames.copy()
    random.shuffle(shuffled_usernames)

    try:
        # Clean up cookies before authentication
        clean_cookies(settings.cookies_file)

        # Ensure authenticated with better cookie management
        if os.path.exists(settings.cookies_file):
            try:
                twikit_client.load_cookies(settings.cookies_file)
                await twikit_client.get_available_locations()
            except Exception as e:
                logger.warning(f"Cookie validation failed: {str(e)}")
                if os.path.exists(settings.cookies_file):
                    os.remove(settings.cookies_file)
                await twikit_client.login(
                    auth_info_1=settings.twitter_username,
                    auth_info_2=settings.twitter_email,
                    password=settings.twitter_password,
                    totp_secret=settings.twitter_totp_secret,
                )
                twikit_client.save_cookies(settings.cookies_file)
        else:
            await twikit_client.login(
                auth_info_1=settings.twitter_username,
                auth_info_2=settings.twitter_email,
                password=settings.twitter_password,
                totp_secret=settings.twitter_totp_secret,
            )
            twikit_client.save_cookies(settings.cookies_file)

        # Fetch tweets for all users with rate limiting
        tasks = [get_cached_tweets(username) for username in shuffled_usernames]
        all_tweets = await asyncio.gather(*tasks)

        # Schedule background refresh for each user
        for username in shuffled_usernames:
            # Get the TTL of the tweets cache
            ttl = await redis.ttl(f"tweets:{username}")
            # Refresh if cache is missing (ttl = -2)
            # or if remaining TTL is less than background_refresh_interval
            if ttl == -2 or (ttl != -1 and ttl < settings.background_refresh_interval):
                background_tasks.add_task(refresh_user_tweets_cache, username)

        logger.info(f"Processed {len(all_tweets)} tweets for {len(shuffled_usernames)} users")

        # Process and filter tweets
        result = {}
        for username, user_tweets in zip(shuffled_usernames, all_tweets):
            result[username] = [
                tweet
                for tweet in user_tweets
                if (include_posts and tweet["type"] == "Post")
                or (include_replies and tweet["type"] == "Reply")
                or (include_retweets and tweet["type"] == "Retweet")
                or (include_quotes and tweet["type"] == "Quote")
            ]

        return result

    except Exception as e:
        logger.error(f"Error in get_tweets: {str(e)}")
        raise


@app.get("/feed.xml")
async def get_feed(
    background_tasks: BackgroundTasks,
    usernames: List[str] = Query(["ylecun", "AndrewYNg", "karpathy", "sama", "geoffreyhinton"]),
    include_posts: bool = True,
    include_replies: bool = True,
    include_retweets: bool = True,
    include_quotes: bool = True,
) -> Response:
    """
    Generate RSS feed for specified Twitter users.

    Args:
        background_tasks: FastAPI background tasks
        usernames: List of Twitter usernames
        include_posts: Whether to include regular posts
        include_replies: Whether to include replies
        include_retweets: Whether to include retweets
        include_quotes: Whether to include quote tweets

    Returns:
        RSS feed response
    """
    fg = FeedGenerator()
    fg.title('The "Totally Not Twitter" Feed')
    fg.description("Your favorite bird site content, now in RSS form!")
    fg.link(href="https://github.com/thytu/XRSS")
    fg.language("en")

    # Add media namespace for profile pictures
    fg.load_extension("media", rss=True)

    tweets_data = await get_tweets(
        background_tasks=background_tasks,
        usernames=usernames,
        include_posts=include_posts,
        include_replies=include_replies,
        include_retweets=include_retweets,
        include_quotes=include_quotes,
    )

    for username, tweets in tweets_data.items():
        # Get user data from cache
        user_data = await get_cached_user(username)

        for tweet in tweets:
            fe = fg.add_entry()
            fe.title(f"{tweet['type']} by {username}")
            fe.link(href=f"https://twitter.com/{username}/status/{tweet['id']}")
            fe.description(tweet["full_text"])
            fe.pubDate(
                datetime.strptime(tweet["created_at"], "%a %b %d %H:%M:%S +0000 %Y").astimezone(
                    ZoneInfo("UTC")
                )
            )
            fe.author({"name": username})
            fe.guid(f"https://twitter.com/{username}/status/{tweet['id']}", permalink=True)

            # Add profile picture as media content if available
            if user_data and user_data.get("profile_image_url"):
                fe.media.content(
                    {
                        "url": user_data["profile_image_url"].replace(
                            "normal", "400x400"
                        ),  # load a higher resolution image
                        "medium": "image",
                        "type": "image/jpeg",
                    }
                )

    return Response(content=fg.rss_str(), media_type="application/rss+xml")


def main() -> None:
    """Entry point for the application."""
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=8003,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
