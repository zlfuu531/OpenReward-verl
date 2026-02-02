#!/usr/bin/env bash
# ------------------------------------------------------------------
# VERL Docker 智能启动脚本
# - 如果容器存在，则直接启动并进入。
# - 如果容器不存在，则自动创建。
# ------------------------------------------------------------------
set -euo pipefail

# ================= 可按需修改 =================
IMAGE_TAG="verlai/verl:vllm012.latest"      # 官方 vLLM 稳定镜像
CONTAINER_NAME="verl-zlf-new"                   # 容器名
HOST_NFS="/nfsdata-117"                     # 宿主机挂载的根目录
CONTAINER_NFS="/nfsdata-117"                # 容器内对应路径
WORKDIR="/nfsdata-117/project/finvlr1/zlf/guotai-reward/verl"  # 容器默认工作目录
SHM_SIZE="32g"                              # 共享内存
# =============================================

# 检查容器是否存在
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  # 容器存在，直接启动
  echo "[1/2] 发现已存在的容器 '${CONTAINER_NAME}'。"

  # 检查容器是否在运行，如果不在则启动它
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "      -> 容器未运行，正在启动..."
    docker start "${CONTAINER_NAME}"
  else
    echo "      -> 容器已在运行中。"
  fi
else
  # 容器不存在，执行完整的创建流程
  echo "[1/3] 未发现容器 '${CONTAINER_NAME}'，将创建新容器。"

  echo "      -> [a] 正在拉取最新镜像: ${IMAGE_TAG}"
  docker pull "${IMAGE_TAG}"

  echo "      -> [b] 正在创建新容器..."
  docker create \
    --runtime=nvidia \
    --gpus all \
    --net=host \
    --shm-size="${SHM_SIZE}" \
    --cap-add=SYS_ADMIN \
    -v "${HOST_NFS}:${CONTAINER_NFS}" \
    -w "${WORKDIR}" \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_TAG}" \
    sleep infinity

  echo "      -> [c] 正在启动新容器..."
  docker start "${CONTAINER_NAME}"
  echo "      -> 容器创建并启动成功。"
fi

# 统一的进入容器步骤
echo "[2/2] 正在进入容器 '${CONTAINER_NAME}'..."
# 如果 WORKDIR 不存在，fallback 到 /nfsdata-117
CMD="if [ -d \"${WORKDIR}\" ]; then cd \"${WORKDIR}\"; else echo '[WARN] 工作目录不存在，切换到 ${CONTAINER_NFS}'; cd \"${CONTAINER_NFS}\"; fi; exec bash"
docker exec -it "${CONTAINER_NAME}" bash -c "${CMD}"

echo "已退出容器。"
