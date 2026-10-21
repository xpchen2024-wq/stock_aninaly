"""Research report seeding with price-consistent target prices.

Target prices are NOT hardcoded: they are derived from the live quote so a
"买入" rating can never carry a target below the current price (which would be
nonsense). Upside per rating:

    买入   +18% ~ +30%
    增持   +8%  ~ +18%
    中性   -3%  ~ +6%
    减持   -18% ~ -8%
    卖出   -30% ~ -18%
"""
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.models import ResearchReport, new_uuid
from app.reports import RATING_UPSIDE, live_prices, target_from_price, upside_for  # noqa: F401


REPORTS = [
    {
        "broker": "中金公司",
        "title": "AI 算力芯片：国产替代进入加速期",
        "content": "随着大模型训练需求爆发，国产 AI 芯片迎来历史性机遇。寒武纪、海光信息在算力密度和软件生态上取得突破，预计 2026 年国产 AI 芯片市场份额将突破 15%。",
        "stock_symbol": "688256.SH",
        "stock_name": "寒武纪",
        "rating": "增持",
        "core_opinion": "国产 AI 芯片替代逻辑加速，看好算力龙头",
    },
    {
        "broker": "中信证券",
        "title": "新能源汽车 2026 中期策略：龙头恒强",
        "content": "比亚迪、宁德时代市占率持续提升，技术壁垒加深。预计全年新能源乘用车销量突破 1200 万辆，行业 CR5 提升至 75%。",
        "stock_symbol": "002594.SZ",
        "stock_name": "比亚迪",
        "rating": "买入",
        "core_opinion": "新能源车渗透率持续提升，龙头估值修复",
    },
    {
        "broker": "华泰证券",
        "title": "光伏行业触底信号已现，左侧布局良机",
        "content": "硅料价格企稳，组件开工率回升至 75%。隆基绿能、通威股份估值处于历史底部区间，建议左侧布局。",
        "stock_symbol": "601012.SH",
        "stock_name": "隆基绿能",
        "rating": "买入",
        "core_opinion": "光伏行业周期触底，龙头估值修复空间大",
    },
    {
        "broker": "国泰君安",
        "title": "半导体设备国产化：刻蚀机率先突破",
        "content": "中微公司、北方华创在 14nm 刻蚀工艺取得验证通过。受益于晶圆厂扩产，设备国产化率有望从 35% 提升至 50%。",
        "stock_symbol": "688012.SH",
        "stock_name": "中微公司",
        "rating": "买入",
        "core_opinion": "半导体设备国产替代加速，刻蚀机龙头受益",
    },
    {
        "broker": "海通证券",
        "title": "银行板块：高股息防御属性凸显",
        "content": "国有大行分红率维持 30%+，股息率 6% 以上。在利率下行周期中，高股息策略具备显著超额收益。",
        "stock_symbol": "601398.SH",
        "stock_name": "工商银行",
        "rating": "增持",
        "core_opinion": "高股息防御策略，银行板块配置价值提升",
    },
    {
        "broker": "招商证券",
        "title": "医药创新：CRO 行业景气度回升",
        "content": "海外投融资回暖，国内创新药管线出海加速。药明康德、康龙化成在手订单同比+25%，业绩拐点已现。",
        "stock_symbol": "603259.SH",
        "stock_name": "药明康德",
        "rating": "买入",
        "core_opinion": "CRO 行业景气度回升，龙头业绩拐点已现",
    },
    {
        "broker": "中金公司",
        "title": "港股互联网：估值修复进行时",
        "content": "腾讯、阿里回购规模合计超 200 亿美元，现金流稳健。AI 商业化加速带来新的增长曲线。",
        "stock_symbol": "00700.HK",
        "stock_name": "腾讯控股",
        "rating": "买入",
        "core_opinion": "港股互联网龙头估值修复，AI 新增长曲线",
    },
    {
        "broker": "中信证券",
        "title": "军工电子：信息化作战装备加速列装",
        "content": "十四五末期军费开支维持高增速，电子对抗、卫星通信等信息化装备订单饱满。",
        "stock_symbol": "002465.SZ",
        "stock_name": "海格通信",
        "rating": "增持",
        "core_opinion": "军工电子信息化装备需求旺盛",
    },
    {
        "broker": "华泰证券",
        "title": "消费电子：AI 手机换机周期启动",
        "content": "Apple Intelligence、华为 HarmonyOS AI 等端侧大模型落地，AI 手机渗透率快速提升。",
        "stock_symbol": "300433.SZ",
        "stock_name": "蓝思科技",
        "rating": "买入",
        "core_opinion": "AI 手机换机周期启动，供应链龙头受益",
    },
    {
        "broker": "国泰君安",
        "title": "煤炭板块：低估值高分红配置机会",
        "content": "动力煤价格企稳，煤企分红率提升至 40-50%。中国神华、陕西煤业股息率 8%+。",
        "stock_symbol": "601088.SH",
        "stock_name": "中国神华",
        "rating": "增持",
        "core_opinion": "煤炭高分红低估值，防御属性强",
    },
    {
        "broker": "海通证券",
        "title": "机器人产业链：特斯拉 Optimus 量产在即",
        "content": "特斯拉 Optimus V3 预计 2026Q4 量产，国内核心供应商有望深度参与。",
        "stock_symbol": "300124.SZ",
        "stock_name": "汇川技术",
        "rating": "买入",
        "core_opinion": "人形机器人量产元年，零部件龙头受益",
    },
    {
        "broker": "招商证券",
        "title": "存储芯片：DDR5 涨价周期开启",
        "content": "AI 推理需求拉动 DDR5、HBM 紧缺，模组价格 Q3 预计上涨 15-20%。",
        "stock_symbol": "603986.SH",
        "stock_name": "兆易创新",
        "rating": "买入",
        "core_opinion": "存储芯片涨价周期开启，国产模组龙头受益",
    },
    {
        "broker": "中金公司",
        "title": "白酒板块：去库存进入尾声",
        "content": "茅台、五粮液批价企稳，渠道库存回归健康水平。中秋国庆双节动销有望超预期。",
        "stock_symbol": "600519.SH",
        "stock_name": "贵州茅台",
        "rating": "增持",
        "core_opinion": "白酒去库存尾声，龙头估值修复",
    },
    {
        "broker": "中信证券",
        "title": "券商板块：并购重组催化估值重估",
        "content": "国君海通合并落地，行业并购重组加速。头部券商 ROE 有望提升至 8-10%。",
        "stock_symbol": "601211.SH",
        "stock_name": "国泰君安",
        "rating": "买入",
        "core_opinion": "券商并购重组催化，头部券商估值重估",
    },
    {
        "broker": "华泰证券",
        "title": "风电海缆：深远海项目订单加速",
        "content": "江苏、广东海风项目招标启动，东方电缆、中天科技 220kV 海缆订单饱满。",
        "stock_symbol": "603606.SH",
        "stock_name": "东方电缆",
        "rating": "买入",
        "core_opinion": "海风项目加速，海缆龙头订单饱满",
    },
    {
        "broker": "国泰君安",
        "title": "家电龙头：出海与以旧换新双轮驱动",
        "content": "海外自有品牌占比提升，空调海外排产同比+20%；国补政策延续支撑内销。",
        "stock_symbol": "000333.SZ",
        "stock_name": "美的集团",
        "rating": "买入",
        "core_opinion": "全球化布局兑现，龙头份额与利润率同步提升",
    },
    {
        "broker": "招商证券",
        "title": "创新药：出海授权持续兑现",
        "content": "多款国产创新药达成海外授权，首付款与里程碑款合计创新高，商业化能力被验证。",
        "stock_symbol": "600276.SH",
        "stock_name": "恒瑞医药",
        "rating": "买入",
        "core_opinion": "创新药国际化进入收获期，估值体系有望重塑",
    },
    {
        "broker": "海通证券",
        "title": "电网设备：特高压与配网改造共振",
        "content": "年度电网投资计划上修，特高压核准节奏提速，配网侧改造订单持续释放。",
        "stock_symbol": "600089.SH",
        "stock_name": "特变电工",
        "rating": "增持",
        "core_opinion": "电网投资上行周期，输变电设备订单饱满",
    },
    {
        "broker": "中信证券",
        "title": "快递物流：件量增速回升，单票成本改善",
        "content": "月度业务量增速回升至两位数，自动化分拣与规模效应推动单票成本下行。",
        "stock_symbol": "002352.SZ",
        "stock_name": "顺丰控股",
        "rating": "增持",
        "core_opinion": "件量与成本双改善，盈利质量提升",
    },
    {
        "broker": "中金公司",
        "title": "基础化工：供给侧出清推动价差修复",
        "content": "新增产能投放放缓，行业开工率回升，MDI/纯碱价差自底部修复。",
        "stock_symbol": "600309.SH",
        "stock_name": "万华化学",
        "rating": "增持",
        "core_opinion": "周期底部反转，龙头成本优势放大利润弹性",
    },
    {
        "broker": "华泰证券",
        "title": "消费建材：地产链出清，龙头集中度提升",
        "content": "地产竣工下行压制需求，行业出清推动份额向头部集中，现金流质量改善。",
        "stock_symbol": "002271.SZ",
        "stock_name": "东方雨虹",
        "rating": "中性",
        "core_opinion": "需求端未见拐点，等待地产链景气修复信号",
    },
    {
        "broker": "国泰君安",
        "title": "种业：转基因商业化推进",
        "content": "转基因玉米大豆产业化试点扩大，品种审定与推广加速，头部种企份额提升。",
        "stock_symbol": "000998.SZ",
        "stock_name": "隆平高科",
        "rating": "增持",
        "core_opinion": "转基因商业化打开新空间，种业龙头受益",
    },
    {
        "broker": "招商证券",
        "title": "机场航空：客流恢复带动免税弹性",
        "content": "国际航线恢复至八成以上，客座率回升，免税租金重谈有望抬升非航收入。",
        "stock_symbol": "600009.SH",
        "stock_name": "上海机场",
        "rating": "中性",
        "core_opinion": "客流修复确定，非航价值重估仍需制度改善",
    },
    {
        "broker": "海通证券",
        "title": "运营商：云业务与分红支撑估值",
        "content": "移动云收入维持双位数增长，资本开支见顶回落，股息率优势凸显。",
        "stock_symbol": "600941.SH",
        "stock_name": "中国移动",
        "rating": "增持",
        "core_opinion": "现金流稳定，高股息资产配置价值凸显",
    },
]


async def _live_prices(symbols) -> Dict[str, float]:
    """Live prices keyed by bare symbol (e.g. 688012)."""
    return await live_prices(symbols)


async def seed_reports(db: AsyncSession, target_count: int = 0):
    """Seed research reports.

    One row per template (no duplicates), published within the last 30 days,
    with target prices derived from live quotes so rating and target agree.
    """
    print(f"📊 Seeding research_reports...")

    await db.execute(delete(ResearchReport))

    now = datetime.utcnow()
    rng = random.Random(42)

    prices = await _live_prices({t["stock_symbol"].split(".")[0] for t in REPORTS})
    print(f"   • live quotes resolved: {len(prices)}/{len(REPORTS)}")

    templates = REPORTS[:target_count] if target_count else REPORTS
    created = 0
    missing_price = 0

    for tmpl in templates:
        bare = tmpl["stock_symbol"].split(".")[0]
        price = prices.get(bare)
        report_id = new_uuid()
        target = target_from_price(price, tmpl["rating"], report_id) if price else None
        if target is None:
            missing_price += 1

        days_ago = rng.randint(0, 29)
        published = now - timedelta(days=days_ago, hours=rng.randint(0, 23))

        db.add(ResearchReport(
            id=report_id,
            broker=tmpl["broker"],
            title=tmpl["title"],
            content=tmpl["content"],
            stock_symbol=tmpl["stock_symbol"],
            stock_name=tmpl["stock_name"],
            rating=tmpl["rating"],
            target_price=target,
            core_opinion=tmpl["core_opinion"],
            published_at=published,
        ))
        created += 1

    await db.commit()
    print(f"   ✓ Inserted {created} research reports (no target price: {missing_price})")

    result = await db.execute(select(ResearchReport))
    total = len(result.scalars().all())
    print(f"   ✓ Database now has {total} reports")
    return created


async def main():
    async with async_session_factory() as db:
        await seed_reports(db)


if __name__ == "__main__":
    asyncio.run(main())
