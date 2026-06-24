"""Seed research_reports table with sample data (IR-001 ~ IR-005)."""
import asyncio
import random
from datetime import datetime, timedelta
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.models import ResearchReport, new_uuid


REPORTS = [
    {
        "broker": "中金公司",
        "title": "AI 算力芯片：国产替代进入加速期",
        "content": "随着大模型训练需求爆发，国产 AI 芯片迎来历史性机遇。寒武纪、海光信息在算力密度和软件生态上取得突破，预计 2026 年国产 AI 芯片市场份额将突破 15%。",
        "stock_symbol": "688256.SH",
        "stock_name": "寒武纪",
        "rating": "增持",
        "target_price": 320.0,
        "core_opinion": "国产 AI 芯片替代逻辑加速，看好算力龙头",
    },
    {
        "broker": "中信证券",
        "title": "新能源汽车 2026 中期策略：龙头恒强",
        "content": "比亚迪、宁德时代市占率持续提升，技术壁垒加深。预计全年新能源乘用车销量突破 1200 万辆，行业 CR5 提升至 75%。",
        "stock_symbol": "002594.SZ",
        "stock_name": "比亚迪",
        "rating": "买入",
        "target_price": 350.0,
        "core_opinion": "新能源车渗透率持续提升，龙头估值修复",
    },
    {
        "broker": "华泰证券",
        "title": "光伏行业触底信号已现，左侧布局良机",
        "content": "硅料价格企稳，组件开工率回升至 75%。隆基绿能、通威股份估值处于历史底部区间，建议左侧布局。",
        "stock_symbol": "601012.SH",
        "stock_name": "隆基绿能",
        "rating": "买入",
        "target_price": 32.0,
        "core_opinion": "光伏行业周期触底，龙头估值修复空间大",
    },
    {
        "broker": "国泰君安",
        "title": "半导体设备国产化：刻蚀机率先突破",
        "content": "中微公司、北方华创在 14nm 刻蚀工艺取得验证通过。受益于晶圆厂扩产，设备国产化率有望从 35% 提升至 50%。",
        "stock_symbol": "688012.SH",
        "stock_name": "中微公司",
        "rating": "买入",
        "target_price": 180.0,
        "core_opinion": "半导体设备国产替代加速，刻蚀机龙头受益",
    },
    {
        "broker": "海通证券",
        "title": "银行板块：高股息防御属性凸显",
        "content": "国有大行分红率维持 30%+，股息率 6% 以上。在利率下行周期中，高股息策略具备显著超额收益。",
        "stock_symbol": "601398.SH",
        "stock_name": "工商银行",
        "rating": "增持",
        "target_price": 8.5,
        "core_opinion": "高股息防御策略，银行板块配置价值提升",
    },
    {
        "broker": "招商证券",
        "title": "医药创新：CRO 行业景气度回升",
        "content": "海外投融资回暖，国内创新药管线出海加速。药明康德、康龙化成在手订单同比+25%，业绩拐点已现。",
        "stock_symbol": "603259.SH",
        "stock_name": "药明康德",
        "rating": "买入",
        "target_price": 95.0,
        "core_opinion": "CRO 行业景气度回升，龙头业绩拐点已现",
    },
    {
        "broker": "中金公司",
        "title": "港股互联网：估值修复进行时",
        "content": "腾讯、阿里回购规模合计超 200 亿美元，现金流稳健。AI 商业化加速带来新的增长曲线。",
        "stock_symbol": "00700.HK",
        "stock_name": "腾讯控股",
        "rating": "买入",
        "target_price": 480.0,
        "core_opinion": "港股互联网龙头估值修复，AI 新增长曲线",
    },
    {
        "broker": "中信证券",
        "title": "军工电子：信息化作战装备加速列装",
        "content": "十四五末期军费开支维持高增速，电子对抗、卫星通信等信息化装备订单饱满。",
        "stock_symbol": "002465.SZ",
        "stock_name": "海格通信",
        "rating": "增持",
        "target_price": 16.0,
        "core_opinion": "军工电子信息化装备需求旺盛",
    },
    {
        "broker": "华泰证券",
        "title": "消费电子：AI 手机换机周期启动",
        "content": "Apple Intelligence、华为 HarmonyOS AI 等端侧大模型落地，AI 手机渗透率快速提升。",
        "stock_symbol": "300433.SZ",
        "stock_name": "蓝思科技",
        "rating": "买入",
        "target_price": 28.0,
        "core_opinion": "AI 手机换机周期启动，供应链龙头受益",
    },
    {
        "broker": "国泰君安",
        "title": "煤炭板块：低估值高分红配置机会",
        "content": "动力煤价格企稳，煤企分红率提升至 40-50%。中国神华、陕西煤业股息率 8%+。",
        "stock_symbol": "601088.SH",
        "stock_name": "中国神华",
        "rating": "增持",
        "target_price": 45.0,
        "core_opinion": "煤炭高分红低估值，防御属性强",
    },
    {
        "broker": "海通证券",
        "title": "机器人产业链：特斯拉 Optimus 量产在即",
        "content": "特斯拉 Optimus V3 预计 2026Q4 量产，国内核心供应商有望深度参与。",
        "stock_symbol": "300124.SZ",
        "stock_name": "汇川技术",
        "rating": "买入",
        "target_price": 85.0,
        "core_opinion": "人形机器人量产元年，零部件龙头受益",
    },
    {
        "broker": "招商证券",
        "title": "存储芯片：DDR5 涨价周期开启",
        "content": "AI 推理需求拉动 DDR5、HBM 紧缺，模组价格 Q3 预计上涨 15-20%。",
        "stock_symbol": "603986.SH",
        "stock_name": "兆易创新",
        "rating": "买入",
        "target_price": 130.0,
        "core_opinion": "存储芯片涨价周期开启，国产模组龙头受益",
    },
    {
        "broker": "中金公司",
        "title": "白酒板块：去库存进入尾声",
        "content": "茅台、五粮液批价企稳，渠道库存回归健康水平。中秋国庆双节动销有望超预期。",
        "stock_symbol": "600519.SH",
        "stock_name": "贵州茅台",
        "rating": "增持",
        "target_price": 1900.0,
        "core_opinion": "白酒去库存尾声，龙头估值修复",
    },
    {
        "broker": "中信证券",
        "title": "券商板块：并购重组催化估值重估",
        "content": "国君海通合并落地，行业并购重组加速。头部券商 ROE 有望提升至 8-10%。",
        "stock_symbol": "601211.SH",
        "stock_name": "国泰君安",
        "rating": "买入",
        "target_price": 22.0,
        "core_opinion": "券商并购重组催化，头部券商估值重估",
    },
    {
        "broker": "华泰证券",
        "title": "风电海缆：深远海项目订单加速",
        "content": "江苏、广东海风项目招标启动，东方电缆、中天科技 220kV 海缆订单饱满。",
        "stock_symbol": "603606.SH",
        "stock_name": "东方电缆",
        "rating": "买入",
        "target_price": 65.0,
        "core_opinion": "海风项目加速，海缆龙头订单饱满",
    },
]


async def seed_reports(db: AsyncSession, target_count: int = 50):
    """Seed research reports: spread across last 30 days, multi-broker."""
    print(f"📊 Seeding research_reports (target={target_count})...")

    # Clear existing
    await db.execute(delete(ResearchReport))

    now = datetime.utcnow()
    created = 0

    # Cycle through templates to reach target_count
    for i in range(target_count):
        tmpl = REPORTS[i % len(REPORTS)]
        # Randomize published_at within last 30 days
        days_ago = random.randint(0, 30)
        published = now - timedelta(days=days_ago, hours=random.randint(0, 23))

        report = ResearchReport(
            id=new_uuid(),
            broker=tmpl["broker"],
            title=tmpl["title"],
            content=tmpl["content"],
            stock_symbol=tmpl["stock_symbol"],
            stock_name=tmpl["stock_name"],
            rating=tmpl["rating"],
            target_price=tmpl["target_price"],
            core_opinion=tmpl["core_opinion"],
            published_at=published,
        )
        db.add(report)
        created += 1

    await db.commit()
    print(f"   ✓ Inserted {created} research reports")

    # Verify
    result = await db.execute(select(ResearchReport))
    total = len(result.scalars().all())
    print(f"   ✓ Database now has {total} reports")
    return created


async def main():
    async with async_session_factory() as db:
        await seed_reports(db)


if __name__ == "__main__":
    asyncio.run(main())
