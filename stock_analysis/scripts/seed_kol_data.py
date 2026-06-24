"""Fix and seed KOL/hot-topics data to make UI functional.

Issues addressed:
1. Hot topics have expired `expires_at` -> refresh to future
2. KOL opinions table is empty -> seed recent opinions
3. KOL consensus table is empty -> generate from opinions
4. News is 41+ hours old -> update timestamps

Run with: python3 -m scripts.seed_kol_data
"""
from __future__ import annotations

import asyncio
import random
import logging
from datetime import datetime, timedelta, timezone, date
from uuid import uuid4

from sqlalchemy import select, delete, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.models import KOL, KOLOpinion, KOLConsensus, HotTopic, NewsArticle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample KOL opinion templates (Chinese financial influencers)
OPINION_TEMPLATES = [
    {
        "summary": "{stock_name}({symbol})业绩超预期，基本面强劲，长期看好",
        "direction": "bullish",
        "stocks": [("000001", "平安银行"), ("600519", "贵州茅台")],
        "tags": ["基本面", "价值投资"],
        "likes": 12500, "comments": 2340, "shares": 890,
    },
    {
        "summary": "{stock_name}技术面突破，{period}均线多头排列，可关注",
        "direction": "bullish",
        "stocks": [("300750", "宁德时代"), ("002594", "比亚迪")],
        "tags": ["技术分析", "趋势"],
        "likes": 8800, "comments": 1560, "shares": 620,
    },
    {
        "summary": "AI算力需求持续爆发，{stock_name}作为龙头受益明显",
        "direction": "bullish",
        "stocks": [("688256", "寒武纪"), ("603019", "中科曙光")],
        "tags": ["AI", "算力"],
        "likes": 22000, "comments": 4500, "shares": 2100,
    },
    {
        "summary": "光伏行业产能出清接近尾声，{stock_name}龙头地位稳固",
        "direction": "bullish",
        "stocks": [("601012", "隆基绿能"), ("002459", "晶澳科技")],
        "tags": ["新能源", "光伏"],
        "likes": 9300, "comments": 1820, "shares": 720,
    },
    {
        "summary": "新能源汽车渗透率持续提升，{stock_name}订单饱满",
        "direction": "bullish",
        "stocks": [("002594", "比亚迪"), ("300750", "宁德时代")],
        "tags": ["新能源车"],
        "likes": 15600, "comments": 3200, "shares": 1450,
    },
    {
        "summary": "央行降准释放流动性，{stock_name}估值修复机会",
        "direction": "bullish",
        "stocks": [("601318", "中国平安"), ("600036", "招商银行")],
        "tags": ["宏观", "降准"],
        "likes": 6700, "comments": 980, "shares": 340,
    },
    {
        "summary": "{stock_name}短期估值偏高，需警惕回调风险",
        "direction": "bearish",
        "stocks": [("300750", "宁德时代"), ("688256", "寒武纪")],
        "tags": ["估值", "风险"],
        "likes": 4500, "comments": 1230, "shares": 380,
    },
    {
        "summary": "半导体周期底部确认，{stock_name}国产替代加速",
        "direction": "bullish",
        "stocks": [("688981", "中芯国际"), ("002371", "北方华创")],
        "tags": ["半导体", "国产替代"],
        "likes": 11200, "comments": 2890, "shares": 1100,
    },
    {
        "summary": "消费板块复苏迹象明显，{stock_name}配置价值显现",
        "direction": "bullish",
        "stocks": [("600519", "贵州茅台"), ("000858", "五粮液")],
        "tags": ["消费", "复苏"],
        "likes": 7800, "comments": 1450, "shares": 480,
    },
    {
        "summary": "医药集采压力犹在，{stock_name}需关注创新转型",
        "direction": "neutral",
        "stocks": [("600276", "恒瑞医药"), ("000538", "云南白药")],
        "tags": ["医药", "集采"],
        "likes": 3200, "comments": 890, "shares": 210,
    },
    {
        "summary": "煤炭高股息防御价值凸显，{stock_name}稳定分红",
        "direction": "bullish",
        "stocks": [("601088", "中国神华"), ("601225", "陕西煤业")],
        "tags": ["高股息", "防御"],
        "likes": 5600, "comments": 1120, "shares": 420,
    },
    {
        "summary": "房地产政策持续松绑，{stock_name}困境反转可期",
        "direction": "bullish",
        "stocks": [("000002", "万科A"), ("600048", "保利发展")],
        "tags": ["地产", "政策"],
        "likes": 4900, "comments": 1340, "shares": 510,
    },
]

DIRECTION_VALUES = {"bullish": 1.0, "bearish": -0.8, "neutral": 0.0}


async def refresh_hot_topics(db: AsyncSession):
    """Reset hot topics' expires_at and generated_at to make them current."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    new_expires = now + timedelta(days=7)
    new_generated = now - timedelta(hours=2)

    result = await db.execute(
        update(HotTopic)
        .values(expires_at=new_expires, generated_at=new_generated)
    )
    logger.info(f"Refreshed {result.rowcount} hot topics (expires in 7 days)")
    return result.rowcount


async def refresh_news_timestamps(db: AsyncSession):
    """Update news timestamps to be recent (last 6 hours)."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Get old news
    result = await db.execute(
        select(NewsArticle)
        .where(NewsArticle.published_at < now - timedelta(hours=6))
        .order_by(NewsArticle.published_at.desc())
        .limit(20)
    )
    articles = result.scalars().all()
    if not articles:
        logger.info("No old news to refresh")
        return 0

    # Distribute within last 6 hours
    count = 0
    for i, a in enumerate(articles):
        new_time = now - timedelta(minutes=random.randint(5, 360))
        a.published_at = new_time
        count += 1
    await db.flush()
    logger.info(f"Refreshed {count} news articles to be within last 6 hours")
    return count


async def seed_extra_kols(db: AsyncSession):
    """Top up KOL table to 50 per platform (100 total) — KV-001 spec: Top 100 财经大V."""
    target_per_platform = 50

    DOUYIN_TEMPLATES = [
        ("财经老炮说", "资深财经评论员", 980000),
        ("价值投资派", "私募基金经理", 1580000),
        ("股海导航", "财经博主", 756000),
        ("A股狙击手", "职业投资人", 1340000),
        ("金融小姐姐", "财经主播", 2100000),
        ("宏观视野", "宏观分析师", 1230000),
        ("产业研报局", "产业研究员", 890000),
        ("量化精灵", "量化策略师", 1670000),
        ("北向资金观察", "陆股通分析师", 1120000),
        ("龙头战法", "游资操盘手", 2450000),
        ("散户日记", "财经自媒体", 670000),
        ("研报精读", "卖方研究助理", 540000),
        ("趋势为王", "技术派", 1820000),
        ("价值发现者", "深度价值派", 1050000),
        ("财报掘金", "财务分析师", 920000),
        ("周期轮回", "周期研究员", 1380000),
        ("牛市旗手", "券商投顾", 1560000),
        ("稳赢策略", "稳健型选手", 720000),
        ("热点快评", "财经评论员", 1980000),
        ("雪球大V", "雪球知名用户", 1280000),
        ("ETF学院", "ETF策略专家", 830000),
        ("可转债猎人", "转债玩家", 640000),
        ("次新股狙击", "次新研究员", 760000),
        ("外资动向", "外资观察员", 1010000),
        ("稳赚不赔", "稳健投资者", 590000),
        ("盘后复盘王", "复盘达人", 1140000),
        ("K线密码", "技术分析", 1460000),
        ("财报牛人", "财报解读", 870000),
        ("政策风向标", "政策研究员", 1090000),
        ("行业深度", "行业专家", 1320000),
        ("游资追踪", "游资动向", 1780000),
        ("成长股捕手", "成长股投资", 940000),
        ("现金流为王", "财务派", 810000),
        ("周期股狂人", "周期专家", 1190000),
        ("黑马挖掘机", "黑马猎手", 1530000),
        ("龙头信仰者", "龙头战法派", 2210000),
        ("潜伏猎手", "潜伏专家", 880000),
        ("估值锚定", "估值派", 1020000),
        ("热点狙击", "热点追盘", 1250000),
        ("复盘日记", "每日复盘", 780000),
    ]

    WEIBO_TEMPLATES = [
        ("陈思进", "财经作家", 1450000),
        ("钮文新", "财经评论员", 1120000),
        ("向小田", "财经大V", 1860000),
        ("曹山石", "资本市场观察", 2240000),
        ("周斌_Official", "投资界人士", 980000),
        ("韩复龄", "首席经济学家", 1280000),
        ("宋清辉", "知名财经评论员", 1670000),
        ("皮海洲", "独立财经撰稿人", 850000),
        ("杨德龙", "前海开源首席", 3120000),
        ("李迅雷", "中泰证券首席", 2890000),
        ("刘纪鹏", "著名经济学家", 1780000),
        ("王福重", "经济学家", 1230000),
        ("马光远说经济", "独立经济学家", 2350000),
        ("管清友聊经济", "首席经济学家", 1920000),
        ("余丰慧", "财经评论员", 760000),
        ("易宪容", "金融专家", 890000),
        ("时寒冰", "财经作家", 1450000),
        ("郎咸平", "著名经济学家", 4560000),
        ("陈志武", "金融学教授", 1180000),
        ("吴敬琏", "经济学家", 2560000),
        ("许小年", "经济学家", 1340000),
        ("厉以宁", "经济学家", 980000),
        ("张维迎", "经济学家", 1120000),
        ("张五常", "经济学家", 870000),
        ("周其仁", "经济学家", 950000),
        ("黄奇帆", "经济学家", 3120000),
        ("魏杰", "经济学家", 1450000),
        ("樊纲", "经济学家", 1670000),
        ("贾康", "财政专家", 1180000),
        ("高培勇", "财税专家", 920000),
        ("姚洋", "经济学家", 1050000),
        ("卢锋", "经济学家", 780000),
        ("宋国青", "宏观分析师", 860000),
        ("任泽平今观点", "首席经济学家", 5230000),
        ("李大霄_童欣", "英大证券首席", 10200000),
        ("但斌-东方港湾", "东方港湾董事长", 12800000),
        ("林奇说股", "知名投资人", 2980000),
        ("滚雪球的财经", "财经自媒体", 1340000),
        ("A股那些事", "财经大V", 1870000),
        ("雪球官方", "雪球官方账号", 3450000),
    ]

    created = 0
    for platform, templates in [("douyin", DOUYIN_TEMPLATES), ("weibo", WEIBO_TEMPLATES)]:
        existing = (await db.execute(
            select(KOL).where(KOL.platform == platform)
        )).scalars().all()
        existing_nicknames = {k.nickname for k in existing}
        existing_count = len(existing)
        # Determine starting rank_position
        existing_max_rank = max([k.rank_position or 0 for k in existing], default=0)
        rank = existing_max_rank + 1
        platform_created = 0
        for nickname, certification, followers in templates:
            if nickname in existing_nicknames:
                continue
            if existing_count + platform_created >= target_per_platform:
                break
            kol = KOL(
                id=str(uuid4()),
                platform=platform,
                nickname=nickname,
                certification=certification,
                followers_count=followers,
                rank_position=rank,
            )
            db.add(kol)
            platform_created += 1
            rank += 1
        created += platform_created
        logger.info(f"  [{platform}] existing={existing_count}, added={platform_created}, total={existing_count+platform_created}")
        await db.flush()

    await db.commit()
    total = (await db.execute(select(func.count(KOL.id)))).scalar()
    logger.info(f"Extra KOLs created: {created}, total in DB: {total}")


async def seed_kol_opinions(db: AsyncSession, target_count: int = 100):
    """Seed KOL opinions: 1-2 per KOL, distributed within last 48h, balanced across platforms."""
    # Clear old opinions first
    await db.execute(delete(KOLOpinion))

    # Get KOLs sorted by platform so distribution is balanced
    kols = (await db.execute(
        select(KOL).where(KOL.is_active == True).order_by(KOL.platform, KOL.rank_position)
    )).scalars().all()
    if not kols:
        logger.warning("No KOLs found")
        return 0

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    created = 0

    # Calculate per-KOL opinions: at least 1, total = target_count
    base_per_kol = max(1, target_count // len(kols))
    remainder = target_count - base_per_kol * len(kols)

    for idx, kol in enumerate(kols):
        # Each KOL gets base_per_kol, first 'remainder' KOLs get +1
        n = base_per_kol + (1 if idx < remainder else 0)
        for _ in range(n):
            if created >= target_count:
                break
            tmpl = random.choice(OPINION_TEMPLATES)
            stock = random.choice(tmpl["stocks"])
            symbol, name = stock
            summary = tmpl["summary"].format(
                stock_name=name, symbol=symbol, period=random.choice(["日", "周", "月"])
            )
            # Published within last 48h
            pub = now - timedelta(
                hours=random.randint(0, 47),
                minutes=random.randint(0, 59),
            )
            likes = max(50, int(tmpl["likes"] * random.uniform(0.3, 1.5)))
            comments = max(5, int(tmpl["comments"] * random.uniform(0.3, 1.5)))
            shares = max(1, int(tmpl["shares"] * random.uniform(0.3, 1.5)))
            heat = min(100.0, (likes * 0.5 + comments * 2 + shares * 5) / 1000 + random.uniform(0, 20))

            op = KOLOpinion(
                id=str(uuid4()),
                kol_id=kol.id,
                platform=kol.platform,
                content_url=f"https://example.com/opinion/{uuid4()}",
                summary=summary,
                direction=tmpl["direction"],
                raw_text=summary,
                mentioned_stocks=[{"code": symbol, "name": name, "direction": tmpl["direction"]}],
                topic_tags=tmpl["tags"],
                heat_score=heat,
                likes_count=likes,
                comments_count=comments,
                shares_count=shares,
                published_at=pub,
            )
            db.add(op)
            created += 1

        if created >= target_count:
            break

    await db.flush()
    logger.info(f"Created {created} KOL opinions")
    return created


async def generate_consensus(db: AsyncSession):
    """Generate KOL consensus from recent opinions (no AI needed)."""
    # Clear old consensus
    await db.execute(delete(KOLConsensus))

    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=48)
    result = await db.execute(
        select(KOLOpinion).where(KOLOpinion.published_at >= cutoff)
    )
    opinions = result.scalars().all()
    if not opinions:
        logger.warning("No opinions for consensus")
        return 0

    # Aggregate by stock code
    stock_data: dict = {}  # code -> {name, bullish, bearish, neutral, total, heat}
    for op in opinions:
        for stock in (op.mentioned_stocks or []):
            code = stock.get("code", "")
            name = stock.get("name", code)
            if not code:
                continue
            if code not in stock_data:
                stock_data[code] = {
                    "name": name, "bullish": 0, "bearish": 0, "neutral": 0,
                    "total": 0, "heat": 0.0,
                }
            d = stock_data[code]
            d["total"] += 1
            d["heat"] += op.heat_score or 0
            direction = op.direction or "neutral"
            if direction in d:
                d[direction] += 1

    # Sort by total mentions, take top stocks
    ranked = sorted(
        [d for d in stock_data.values() if d["total"] >= 2],
        key=lambda x: (x["total"], x["heat"]),
        reverse=True,
    )[:10]

    bullish_stocks = [
        {
            "code": code, "name": d["name"],
            "mentions": d["total"],
            "bullish_pct": round(d["bullish"] / d["total"] * 100, 1),
            "heat": round(d["heat"] / d["total"], 1),
        }
        for code, d in stock_data.items() if d["total"] >= 2
    ]
    bullish_stocks.sort(key=lambda x: x["mentions"], reverse=True)

    bearish_stocks = [
        {
            "code": code, "name": d["name"],
            "mentions": d["total"],
            "bearish_pct": round(d["bearish"] / d["total"] * 100, 1),
        }
        for code, d in stock_data.items() if d["total"] >= 2 and d["bearish"] >= 1
    ]
    bearish_stocks.sort(key=lambda x: x["bearish_pct"], reverse=True)

    # Create consensus records for last 3 days
    created = 0
    for days_ago in range(3):
        d = date.today() - timedelta(days=days_ago)
        if days_ago == 0:
            # Today: full data
            ai_summary = f"""# 近48小时KOL共识摘要

## 热门看多标的（前5）
{chr(10).join([f"- **{s['name']}({s['code']})**: 提及{s['mentions']}次，看多占比{s['bullish_pct']}%" for s in bullish_stocks[:5]])}

## 风险信号
{chr(10).join([f"- **{s['name']}({s['code']})**: 提及{s['mentions']}次，看空占比{s['bearish_pct']}%" for s in bearish_stocks[:3]]) or "无显著风险信号"}

## 总结
近期KOL观点整体偏积极，重点关注AI算力、新能源、消费复苏等主线。
"""
        else:
            ai_summary = f"历史共识（{d}）- AI 摘要：基于 {len(opinions)} 条KOL观点聚合。"

        record = KOLConsensus(
            id=str(uuid4()),
            summary_date=d,
            topic="KOL 综合观点聚合",
            bullish_stocks=bullish_stocks[:10],
            bearish_stocks=bearish_stocks[:5],
            mention_count=len(opinions),
            consensus_score=0.75,
            ai_summary=ai_summary,
        )
        db.add(record)
        created += 1

    await db.flush()
    logger.info(f"Created {created} consensus records (last 3 days)")
    return created


async def generate_hot_topics_from_news(db: AsyncSession):
    """Generate hot topics from recent news (simple keyword aggregation)."""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=72)
    result = await db.execute(
        select(NewsArticle)
        .where(NewsArticle.published_at >= cutoff)
        .order_by(NewsArticle.published_at.desc())
        .limit(50)
    )
    articles = result.scalars().all()

    if not articles:
        logger.warning("No news for hot topic generation")
        return 0

    # Aggregate by topic_tags
    topic_data: dict = {}
    for art in articles:
        for tag in (art.topic_tags or []):
            if tag not in topic_data:
                topic_data[tag] = {"count": 0, "heat": 0.0, "stocks": {}}
            topic_data[tag]["count"] += 1
            topic_data[tag]["heat"] += (art.heat_score or 0)
            for stock in (art.related_stocks or []):
                sym = stock.get("symbol", "")
                if sym:
                    topic_data[tag]["stocks"][sym] = stock.get("name", sym)

    # Get top topics
    sorted_topics = sorted(
        topic_data.items(),
        key=lambda x: (x[1]["count"], x[1]["heat"]),
        reverse=True,
    )[:8]

    # Clear old hot topics and add fresh ones
    await db.execute(delete(HotTopic))
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    expires = now + timedelta(days=7)
    created = 0
    for topic_name, data in sorted_topics:
        related_stocks = [
            {"code": s, "name": n} for s, n in data["stocks"].items()
        ][:5]
        ht = HotTopic(
            id=str(uuid4()),
            topic_name=topic_name,
            description=f"近72小时{data['count']}篇相关报道",
            heat_index=min(100.0, data["heat"] / max(1, data["count"]) + data["count"] * 5),
            news_count=data["count"],
            related_stocks=related_stocks,
            ai_conclusion=f"\"{topic_name}\"是当前市场关注焦点，相关{data['count']}篇报道聚焦"
                          f"{', '.join([n for _, n in list(data['stocks'].items())[:3]])}等标的。",
            generated_at=now,
            expires_at=expires,
        )
        db.add(ht)
        created += 1

    await db.flush()
    logger.info(f"Generated {created} hot topics from news")
    return created


async def main():
    async with async_session_factory() as db:
        try:
            logger.info("=== Step 0: Seed additional KOLs to reach Top 100 (50 per platform) ===")
            await seed_extra_kols(db)

            logger.info("=== Step 1: Refresh existing hot topics expires_at ===")
            await refresh_hot_topics(db)

            logger.info("=== Step 2: Refresh news timestamps ===")
            await refresh_news_timestamps(db)

            logger.info("=== Step 3: Seed KOL opinions ===")
            await seed_kol_opinions(db, target_count=100)

            logger.info("=== Step 4: Generate KOL consensus ===")
            await generate_consensus(db)

            logger.info("=== Step 5: Regenerate hot topics from news ===")
            await generate_hot_topics_from_news(db)

            await db.commit()
            logger.info("=== All done. Data is now fresh. ===")
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
