#!/usr/bin/env bash
# ============================================================================
# AI 股票分析平台 - 本地一键启动 / 停止 / 状态检查
# ============================================================================
# 用法：
#   ./scripts/localctl.sh start     启动全部服务（API + Worker + Beat + 容器）
#   ./scripts/localctl.sh stop      停止全部服务
#   ./scripts/localctl.sh restart   重启
#   ./scripts/localctl.sh status    查看运行状态
#
# 说明：
#   1) 用 nohup + PID 文件管理，进程不依赖终端/IDE 会话，关闭终端也不受影响；
#   2) API 不使用 --reload（--reload 会把一个进程拆成父子两个，父进程退出后
#      子进程会变成占着端口的孤儿进程，难以管理）。改代码后请执行 restart；
#   3) 日志统一写到 logs/ 下，PID 写到 .run/ 下。
# ============================================================================
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PY="$PROJECT_DIR/.venv/bin/python"
RUN_DIR="$PROJECT_DIR/.run"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

API_PORT="${API_PORT:-8000}"
PY="$VENV_PY"

c_red()   { echo -e "\033[31m$1\033[0m"; }
c_green() { echo -e "\033[32m$1\033[0m"; }
c_yellow(){ echo -e "\033[33m$1\033[0m"; }
c_cyan()  { echo -e "\033[36m$1\033[0m"; }

# -- 基础检查 ---------------------------------------------------------------
check_env() {
  if [ ! -x "$PY" ]; then
    c_red "✗ 未找到虚拟环境: $PY"
    echo "  请先创建: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
  fi
  if ! command -v docker >/dev/null 2>&1; then
    c_yellow "! 未检测到 docker，PostgreSQL/Redis 可能未运行"
  fi
}

# -- 容器 -------------------------------------------------------------------
start_containers() {
  if command -v docker >/dev/null 2>&1; then
    c_cyan "▸ 启动数据库与缓存容器..."
    docker compose up -d postgres redis 2>/dev/null \
      || docker start stock_postgres stock_redis 2>/dev/null \
      || c_yellow "! 容器启动失败，请手动检查 OrbStack/Docker"
    sleep 2
  fi
}

# -- 进程管理 ---------------------------------------------------------------
pid_of() {  # $1 = 名称
  local f="$RUN_DIR/$1.pid"
  [ -f "$f" ] && cat "$f" || echo ""
}

is_alive() {  # $1 = pid
  [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null
}

stop_one() {  # $1 = 名称
  local name="$1" pid
  pid="$(pid_of "$name")"
  if is_alive "$pid"; then
    kill "$pid" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      is_alive "$pid" || break
      sleep 0.3
    done
    is_alive "$pid" && kill -9 "$pid" 2>/dev/null
    c_yellow "  已停止 $name (pid $pid)"
  fi
  rm -f "$RUN_DIR/$name.pid"
}

start_one() {  # $1 = 名称, $2.. = 命令
  local name="$1"; shift
  local pid; pid="$(pid_of "$name")"
  if is_alive "$pid"; then
    c_green "  ● $name 已在运行 (pid $pid)"
    return 0
  fi
  nohup "$@" >> "$LOG_DIR/$name.log" 2>&1 &
  echo $! > "$RUN_DIR/$name.pid"
  sleep 0.6
  local newpid; newpid="$(pid_of "$name")"
  if is_alive "$newpid"; then
    c_green "  ✔ $name 已启动 (pid $newpid)"
  else
    c_red "  ✗ $name 启动失败，详见 $LOG_DIR/$name.log"
    return 1
  fi
}

start_all() {
  check_env
  echo ""
  c_cyan "▶ 启动 AI 股票分析平台"
  start_containers

  # Celery Beat 必须先停，否则会与旧调度器重复投递
  stop_one beat

  start_one api    "$PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT" --workers 1
  start_one worker "$PY" -m celery -A app.celery_app worker --loglevel=info -Q crawler,analysis,maintenance,data --concurrency=4
  start_one beat   "$PY" -m celery -A app.celery_app beat --loglevel=info

  echo ""
  sleep 2
  status_all
  echo ""
  c_cyan "▸ 访问地址："
  echo "   前端主页   http://localhost:$API_PORT/ui/"
  echo "   API 文档   http://localhost:$API_PORT/docs"
  echo "   健康检查   http://localhost:$API_PORT/health"
  echo "   默认账号   admin / admin123"
  echo ""
}

stop_all() {
  echo ""
  c_cyan "■ 停止服务"
  stop_one beat
  stop_one worker
  stop_one api
  # 兜底：清掉可能残留的 8000 端口占用（如 --reload 遗留的孤儿进程）
  local leftover
  leftover="$(lsof -t -iTCP:"$API_PORT" -sTCP:LISTEN 2>/dev/null | head -5)"
  if [ -n "$leftover" ]; then
    echo "$leftover" | xargs kill -9 2>/dev/null
    c_yellow "  已清理残留端口占用"
  fi
  echo ""
}

status_all() {
  echo ""
  c_cyan "■ 运行状态"
  for name in api worker beat; do
    local pid; pid="$(pid_of "$name")"
    if is_alive "$pid"; then
      c_green "  ● $name 运行中 (pid $pid)"
    else
      c_red "  ○ $name 未运行"
    fi
  done
  if curl -s -m 3 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    c_green "  ● HTTP 健康检查通过  http://localhost:$API_PORT/health"
  else
    c_yellow "  ○ HTTP 未响应（API 未启动或仍在初始化）"
  fi
  if command -v docker >/dev/null 2>&1; then
    docker ps --filter "name=stock_" --format "  · {{.Names}} {{.Status}}" 2>/dev/null
  fi
  echo ""
}

case "${1:-status}" in
  start)   start_all ;;
  stop)    stop_all ;;
  restart) stop_all; start_all ;;
  status)  status_all ;;
  *)       echo "用法: $0 {start|stop|restart|status}"; exit 1 ;;
esac
