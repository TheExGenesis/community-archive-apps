# community archive twitter bot

nostalgia‑drip daily tweets highlighting bangers from the community archive.

check it on twitter: https://x.com/ca_highlights

---

## tl;dr

- **runs every 15 min via val.town**
- builds a queue of tweets once per utc‑day (00‑**:**)
- posts each item at the _same hour & minute_ the original tweet dropped, give or take the 15 min tick
- uses **supabase** for archive lookup, **sqlite** (val.fs) for local state, and **oauth 1.0a** creds to hit `POST /2/tweets`

---

## anatomy

| layer                        | purpose                                                                             |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `communityArchiveBot.ts`     | cron handler, queue builder, queue poster                                           |
| `to_post_queue` (sqlite)     | rows waiting to go live — `tweet_id`, `text`, `scheduled_at`                        |
| `posted_tweets` (sqlite)     | permanent log of everything we already surfaced (plus daily dummy rows)             |
| `enriched_tweets` (supabase) | raw firehose of historical tweets (must include `username`, `favorite_count`, etc.) |
| `all_account` (supabase)     | whitelist of public accounts we'll resurface                                        |

---

## workflow

1. **every run (\*/15 min):** `cronValHandler()` executes.
2. **queue build step (once/day):**
   - if nothing queued for today _and_ no “dummy” row in `posted_tweets`, call `selectHighlights()`.
   - result filtered & capped: 3‑12 tweets, ≤2 per user/year, likes ≥100.
   - craft new tweet text (truncates to fit, adds credit + link).
   - for each, calculate today’s `scheduled_at` matching original `HH:MM` (utc) and insert into `to_post_queue`.
   - if zero highlights, insert `dummy-YYYY-MM-DD` into `posted_tweets` to mark the attempt.
3. **posting step:**
   - grab any queue rows with `scheduled_at ≤ now + 15 min`.
   - `postTweet()` signs the request with oauth1 and fires to x api v2.
   - on success, move row to `posted_tweets`.

---

## deployment

1. **fork or create** a val named `communityArchiveBot`.
2. add secrets (val → **secrets** pane):
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE`
   - `X_API_KEY`, `X_API_SECRET` (consumer key/secret)
   - `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET` (user token/secret)
   - optional `X_USERNAME` (bot’s handle for nicer logs)
3. set schedule to `*/15 * * * *` (val cron expression).
4. ship it.

> **note** : oauth2 user‑bearer works too, but basic tier is happy with oauth1 and the four tokens are easier to yank from the dev portal.

---

## supabase expectations

```sql
create table enriched_tweets (
  tweet_id        text primary key,
  account_id      text,
  username        text,
  created_at      timestamptz,
  full_text       text,
  favorite_count  int,
  reply_to_tweet_id text
);

create table all_account (
  account_id text primary key
);
```

no writes are performed on supabase — it’s read‑only.

---

## hacking / local test

```ts
// peek at upcoming queue
await previewQueue();

// manual fire
await cronValHandler({ lastRunAt: undefined });
```

---
