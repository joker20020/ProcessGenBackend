#!/usr/bin/env python3
"""
使用 aiohttp 异步请求图像生成 API 的示例

运行方式:
    uv run python examples/aiohttp_image_generation.py
"""

import asyncio
import json
from pathlib import Path
from io import BytesIO

import aiohttp
from PIL import Image


BASE_URL = "http://localhost:8050"
WORKFLOW_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "workflow"
    / "Flux-Dev-ComfyUI-Workflow.json"
)


def load_flux_workflow() -> dict:
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


async def text_to_image(
    session: aiohttp.ClientSession,
    prompt: str,
    output_path: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    seed: int | None = None,
    cfg_scale: float = 3.5,
    sampler_name: str = "dpmpp_2m",
    scheduler: str = "simple",
    checkpoint: str = "flux1-dev.safetensors",
    loras: list[dict] | None = None,
) -> bool:
    """
    文生图请求

    Args:
        session: aiohttp 会话
        prompt: 正向提示词
        output_path: 输出图像路径
        negative_prompt: 负向提示词
        width: 图像宽度
        height: 图像高度
        steps: 采样步数
        seed: 随机种子
        cfg_scale: CFG 引导强度
        sampler_name: 采样器名称
        scheduler: 调度器名称
        checkpoint: 模型检查点
        loras: LoRA 列表

    Returns:
        是否成功
    """
    workflow = load_flux_workflow()

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "checkpoint": checkpoint,
        "workflow": workflow,
    }

    if seed is not None:
        payload["seed"] = seed

    if loras:
        payload["loras"] = loras

    print(f"\n[Text-to-Image] 正在生成: {prompt[:50]}...")

    try:
        async with session.post(
            f"{BASE_URL}/api/v1/text-to-image",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as response:
            if response.status == 200:
                image_data = await response.read()
                with open(output_path, "wb") as f:
                    f.write(image_data)
                print(f"[Text-to-Image] 成功! 保存至: {output_path}")
                return True
            else:
                error = await response.text()
                print(f"[Text-to-Image] 失败: {response.status} - {error}")
                return False
    except asyncio.TimeoutError:
        print("[Text-to-Image] 超时!")
        return False
    except Exception as e:
        print(f"[Text-to-Image] 错误: {e}")
        return False


async def image_to_image(
    session: aiohttp.ClientSession,
    prompt: str,
    init_image_path: str,
    output_path: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    seed: int | None = None,
    cfg_scale: float = 3.5,
    sampler_name: str = "dpmpp_2m",
    scheduler: str = "simple",
    checkpoint: str = "flux1-dev.safetensors",
    strength: float = 0.75,
    loras: list[dict] | None = None,
) -> bool:
    """
    图生图请求

    Args:
        session: aiohttp 会话
        prompt: 正向提示词
        init_image_path: 初始图像路径
        output_path: 输出图像路径
        negative_prompt: 负向提示词
        width: 图像宽度
        height: 图像高度
        steps: 采样步数
        seed: 随机种子
        cfg_scale: CFG 引导强度
        sampler_name: 采样器名称
        scheduler: 调度器名称
        checkpoint: 模型检查点
        strength: 图生图强度
        loras: LoRA 列表

    Returns:
        是否成功
    """
    workflow = load_flux_workflow()

    with open(init_image_path, "rb") as f:
        init_image_data = f.read()

    data = aiohttp.FormData()
    data.add_field("prompt", prompt)
    data.add_field("negative_prompt", negative_prompt)
    data.add_field("width", str(width))
    data.add_field("height", str(height))
    data.add_field("steps", str(steps))
    data.add_field("cfg_scale", str(cfg_scale))
    data.add_field("sampler_name", sampler_name)
    data.add_field("scheduler", scheduler)
    data.add_field("checkpoint", checkpoint)
    data.add_field("strength", str(strength))
    data.add_field("workflow", json.dumps(workflow))
    data.add_field(
        "init_image",
        init_image_data,
        filename=Path(init_image_path).name,
        content_type="image/png",
    )

    if seed is not None:
        data.add_field("seed", str(seed))

    if loras:
        data.add_field("loras", json.dumps(loras))

    print(f"\n[Image-to-Image] 正在处理: {prompt[:50]}...")

    try:
        async with session.post(
            f"{BASE_URL}/api/v1/image-to-image",
            data=data,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as response:
            if response.status == 200:
                image_data = await response.read()
                with open(output_path, "wb") as f:
                    f.write(image_data)
                print(f"[Image-to-Image] 成功! 保存至: {output_path}")
                return True
            else:
                error = await response.text()
                print(f"[Image-to-Image] 失败: {response.status} - {error}")
                return False
    except asyncio.TimeoutError:
        print("[Image-to-Image] 超时!")
        return False
    except Exception as e:
        print(f"[Image-to-Image] 错误: {e}")
        return False


async def batch_text_to_image(
    prompts: list[str],
    output_dir: str = "output",
    **kwargs,
) -> list[str]:
    """
    批量文生图请求（并发执行）

    Args:
        prompts: 提示词列表
        output_dir: 输出目录
        **kwargs: 传递给 text_to_image 的其他参数

    Returns:
        成功生成的图像路径列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, prompt in enumerate(prompts):
            task = text_to_image(
                session=session,
                prompt=prompt,
                output_path=str(output_path / f"generated_{i + 1}.png"),
                **kwargs,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    successful = [
        str(output_path / f"generated_{i + 1}.png")
        for i, success in enumerate(results)
        if success
    ]
    print(f"\n批量生成完成: {len(successful)}/{len(prompts)} 成功")
    return successful


async def health_check(session: aiohttp.ClientSession) -> dict | None:
    """健康检查"""
    try:
        async with session.get(f"{BASE_URL}/health") as response:
            if response.status == 200:
                return await response.json()
    except Exception as e:
        print(f"健康检查失败: {e}")
    return None


async def main():
    print("=" * 60)
    print("aiohttp 图像生成示例")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # 健康检查
        print("\n1. 健康检查...")
        health = await health_check(session)
        if health:
            print(f"   状态: {health['status']}")
            print(f"   嵌入模型: {health['embedding_model_name']}")
            print(f"   重排模型: {health['rerank_model_name']}")
        else:
            print("   服务不可用，请确保 API 服务正在运行")
            return

        # 单次文生图
        print("\n2. 文生图测试...")
        await text_to_image(
            session=session,
            prompt="a beautiful fantasy game card with magical elements",
            output_path="output/example_text2img.png",
            width=1024,
            height=1024,
            steps=20,
            cfg_scale=3.5,
        )

        # 带 LoRA 的文生图
        print("\n3. 文生图 + LoRA 测试...")
        await text_to_image(
            session=session,
            prompt="a detailed character portrait for a game card",
            output_path="output/example_text2img_lora.png",
            loras=[
                {"name": "detail_tweaker.safetensors", "strength": 0.8},
            ],
        )

        # 图生图（需要先有输入图像）
        print("\n4. 图生图测试...")
        init_img_path = Path("output/example_text2img.png")
        if init_img_path.exists():
            await image_to_image(
                session=session,
                prompt="transform into a dark fantasy style",
                init_image_path=str(init_img_path),
                output_path="output/example_img2img.png",
                strength=0.6,
            )
        else:
            print("   跳过: 需要先运行文生图生成初始图像")

    # 批量生成
    print("\n5. 批量文生图测试...")
    prompts = [
        "a warrior game card with golden armor",
        "a mage game card with fire magic",
        "a rogue game card with shadow abilities",
    ]
    await batch_text_to_image(
        prompts=prompts,
        output_dir="output/batch",
        width=1024,
        height=1024,
        steps=20,
    )

    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
