// communityArchiveBot.ts – single cronValHandler every 15 min
// -----------------------------------------------------------------------------
// logic each run:
// 1. if we've never queued for *today*, build queue via selectHighlights().
//    ‑ if no highlights, insert sentinel row into posted_tweets (tweet_id = "dummy‑YYYY‑MM‑DD").
// 2. fetch any queued tweets scheduled within the next 15 min window and post them.
// -----------------------------------------------------------------------------
// env secrets:
//   NEXT_PUBLIC_SUPABASE_URL, SUPABASE_SERVICE_ROLE
//   X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
//   X_USERNAME (optional label)
// -----------------------------------------------------------------------------
// sqlite tables (autocreated):
//   posted_tweets(tweet_id PRIMARY KEY, posted_at TEXT)
//   to_post_queue(tweet_id PRIMARY KEY, text TEXT, scheduled_at TEXT)
// -----------------------------------------------------------------------------
// schedule: set this val to run */15 * * * *  (every 15 min)

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import CryptoJS from "https://esm.sh/crypto-js@4.2.0?bundle";
import OAuth from "https://esm.sh/oauth-1.0a@2.2.6?bundle";
import { sqlite } from "https://esm.town/v/std/sqlite";

// ---------- setup ----------
const supabase = createClient(
  Deno.env.get("NEXT_PUBLIC_SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE")!
);

const apiKey = Deno.env.get("X_API_KEY")!;
const apiSecret = Deno.env.get("X_API_SECRET")!;
const userToken = Deno.env.get("X_ACCESS_TOKEN")!;
const userSecret = Deno.env.get("X_ACCESS_TOKEN_SECRET")!;
const screen = Deno.env.get("X_USERNAME") ?? "communityArchive";

// create tables if absent
await sqlite.execute(`create table if not exists posted_tweets(
  tweet_id text primary key,
  posted_at text
)`);
await sqlite.execute(`create table if not exists to_post_queue(
  tweet_id text primary key,
  text text,
  scheduled_at text
)`);

// ---------- oauth1 tweet helper ----------
const oauth = new OAuth({
  consumer: { key: apiKey, secret: apiSecret },
  signature_method: "HMAC-SHA1",
  hash_function(base, key) {
    return CryptoJS.HmacSHA1(base, key).toString(CryptoJS.enc.Base64);
  },
});
async function postTweet(text: string): Promise<boolean> {
  const url = "https://api.twitter.com/2/tweets";
  const body = JSON.stringify({ text });
  const authHeader = oauth.toHeader(
    oauth.authorize(
      { url, method: "POST" },
      { key: userToken, secret: userSecret }
    )
  );
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", ...authHeader },
    body,
  });
  if (!res.ok) {
    console.error("tweet error", res.status, await res.text(), {
      text,
      authHeader,
    });
    return false;
  }
  const j = await res.json();
  console.log(
    "tweeted →",
    `https://twitter.com/${screen}/status/${j.data?.id ?? ""}`
  );
  return true;
}

// ---------- highlight selection (unchanged) ----------
async function selectHighlights(today = new Date()) {
  const m = today.getUTCMonth() + 1;
  const d = today.getUTCDate();
  const mStr = String(m).padStart(2, "0");
  const dStr = String(d).padStart(2, "0");
  const thisYear = today.getUTCFullYear();

  const { data: accs } = await supabase
    .from("all_account")
    .select("account_id");
  const publicIds = new Set((accs ?? []).map((a: any) => a.account_id));

  const postedIds = new Set(
    (await sqlite.execute("select tweet_id from posted_tweets")).rows.map(
      (r: any) => r[0]
    )
  );
  const queuedIds = new Set(
    (await sqlite.execute("select tweet_id from to_post_queue")).rows.map(
      (r: any) => r[0]
    )
  );

  const pageSize = 1000;
  const candidates: any[] = [];

  for (let yr = thisYear - 1; yr >= 2006; yr--) {
    const start = `${yr}-${mStr}-${dStr}T00:00:00Z`;
    const end = new Date(Date.UTC(yr, m - 1, d) + 86_400_000).toISOString();

    for (let page = 0; ; page++) {
      const { data } = await supabase
        .from("enriched_tweets")
        .select(
          "tweet_id,account_id,username,created_at,full_text,favorite_count,reply_to_tweet_id"
        )
        .gte("created_at", start)
        .lt("created_at", end)
        .range(page * pageSize, page * pageSize + pageSize - 1);
      if (!data?.length) break;

      for (const t of data) {
        if (postedIds.has(t.tweet_id) || queuedIds.has(t.tweet_id)) continue;
        if (!publicIds.has(t.account_id)) continue;
        if (t.full_text?.startsWith("RT @")) continue;
        if (t.reply_to_tweet_id) continue;
        if ((t.favorite_count ?? 0) < 100) continue;
        candidates.push(t);
      }
      if (data.length < pageSize) break;
    }
  }
  if (!candidates.length) return [];

  candidates.sort((a, b) => (b.favorite_count ?? 0) - (a.favorite_count ?? 0));
  const perUser: Record<string, number> = {},
    perYear: Record<number, number> = {},
    final: any[] = [];
  for (const t of candidates) {
    const yr = new Date(t.created_at).getUTCFullYear();
    if ((perYear[yr] ?? 0) >= 2) continue;
    if ((perUser[t.account_id] ?? 0) >= 2) continue;
    perYear[yr] = (perYear[yr] ?? 0) + 1;
    perUser[t.account_id] = (perUser[t.account_id] ?? 0) + 1;
    final.push(t);
    if (final.length >= 12) break;
  }
  return final;
}

// ---------- compose tweet text ----------
function craftText(t: any): string {
  const yr = new Date(t.created_at).getUTCFullYear();
  const pre = `on this day in ${yr}, @${t.username} tweeted:\n\n`;
  const reserve = 24;
  const url = `https://twitter.com/${t.username}/status/${t.tweet_id}`;
  return `${pre}\n${url}`;
}

// ---------- main cron handler ----------
export interface Interval {
  lastRunAt: Date | undefined;
}
export default async function cronValHandler(interval: Interval) {
  console.log("Cron val ran!");
  const now = new Date();
  const todayStr = now.toISOString().slice(0, 10); // YYYY‑MM‑DD (UTC)

  // 1. ensure queue built for today (or dummy)
  const queuedToday =
    (
      await sqlite.execute({
        sql: "select 1 from to_post_queue where scheduled_at like ? limit 1",
        args: [`${todayStr}%`],
      })
    ).rows.length > 0;
  const dummyExists =
    (
      await sqlite.execute({
        sql: "select 1 from posted_tweets where tweet_id = ? limit 1",
        args: [`dummy-${todayStr}`],
      })
    ).rows.length > 0;

  if (!queuedToday && !dummyExists) {
    const highlights = await selectHighlights(now);
    if (highlights.length === 0) {
      console.log("no highlights – recording dummy for", todayStr);
      await sqlite.execute({
        sql: "insert or ignore into posted_tweets(tweet_id, posted_at) values(?, datetime('now'))",
        args: [`dummy-${todayStr}`],
      });
    } else {
      console.log(`queuing ${highlights.length} highlights for`, todayStr);
      for (const t of highlights) {
        const created = new Date(t.created_at);
        const sched = new Date(
          Date.UTC(
            now.getUTCFullYear(),
            now.getUTCMonth(),
            now.getUTCDate(),
            created.getUTCHours(),
            created.getUTCHours() === 23 && created.getUTCMinutes() === 59
              ? 58
              : created.getUTCMinutes(),
            0
          )
        );
        await sqlite.execute({
          sql: "insert or ignore into to_post_queue(tweet_id,text,scheduled_at) values(?,?,?)",
          args: [t.tweet_id, craftText(t), sched.toISOString()],
        });
      }
    }
  }

  // 2. post tweets whose scheduled_at ≤ now+15m
  const windowEnd = new Date(now.getTime() + 15 * 60 * 1000).toISOString();
  const due = await sqlite.execute({
    sql: "select tweet_id,text from to_post_queue where scheduled_at <= ? order by scheduled_at",
    args: [windowEnd],
  });
  for (const row of due.rows) {
    const [tweet_id, text] = row;
    if (await postTweet(text)) {
      await sqlite.execute({
        sql: "delete from to_post_queue where tweet_id = ?",
        args: [tweet_id],
      });
      await sqlite.execute({
        sql: "insert or ignore into posted_tweets(tweet_id, posted_at) values(?, datetime('now'))",
        args: [tweet_id],
      });
      await new Promise((r) => setTimeout(r, 15000));
    }
  }
}
