import asyncio
import random
import logging
import uuid
from io import BytesIO
from typing import Optional, Dict, Any

import httpx
from PIL import Image

from config import config
from models import TextToImageRequest, ImageToImageRequest, LoraInfo

logger = logging.getLogger(__name__)


class ComfyUIService:
    _instance: Optional["ComfyUIService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self.base_url = config.comfyui_url.rstrip("/")
            self.timeout = config.comfyui_timeout
            self.client_id = str(uuid.uuid4())
            self._initialized = True

    async def _get_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
        )

    async def upload_image(self, image: Image.Image, filename: str) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        async with await self._get_client() as client:
            files = {"image": (filename, buffer, "image/png")}
            data = {"overwrite": "true"}
            response = await client.post("/upload/image", files=files, data=data)
            response.raise_for_status()
            result = response.json()
            return result.get("name", filename)

    def _find_nodes_by_type(
        self, workflow: Dict[str, Any], node_type: str
    ) -> list[tuple[str, Dict[str, Any]]]:
        nodes = []
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict) and node_data.get("class_type") == node_type:
                nodes.append((node_id, node_data))
        return nodes

    def _inject_parameters_to_workflow(
        self,
        workflow: Dict[str, Any],
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        cfg_scale: float,
        sampler_name: Optional[str],
        scheduler: Optional[str],
        checkpoint: str,
    ) -> Dict[str, Any]:
        workflow = workflow.copy()

        for node_id, node_data in self._find_nodes_by_type(workflow, "KSampler"):
            inputs = node_data.get("inputs", {})
            inputs["seed"] = seed
            inputs["steps"] = steps
            inputs["cfg"] = cfg_scale
            if sampler_name:
                inputs["sampler_name"] = sampler_name
            if scheduler:
                inputs["scheduler"] = scheduler
            node_data["inputs"] = inputs

        for node_id, node_data in self._find_nodes_by_type(
            workflow, "SamplerCustomAdvanced"
        ):
            pass

        for node_id, node_data in self._find_nodes_by_type(workflow, "KSamplerSelect"):
            if sampler_name:
                node_data["inputs"]["sampler_name"] = sampler_name

        for node_id, node_data in self._find_nodes_by_type(workflow, "BasicScheduler"):
            inputs = node_data.get("inputs", {})
            inputs["steps"] = steps
            if scheduler:
                inputs["scheduler"] = scheduler
            node_data["inputs"] = inputs

        for node_id, node_data in self._find_nodes_by_type(workflow, "RandomNoise"):
            node_data["inputs"]["noise_seed"] = seed

        for node_id, node_data in self._find_nodes_by_type(workflow, "FluxGuidance"):
            node_data["inputs"]["guidance"] = cfg_scale

        for node_id, node_data in self._find_nodes_by_type(workflow, "BasicGuider"):
            pass

        # clip_text_nodes = self._find_nodes_by_type(workflow, "CLIPTextEncode")
        # if len(clip_text_nodes) >= 1:
        #     clip_text_nodes[0][1]["inputs"]["text"] = prompt
        # if len(clip_text_nodes) >= 2:
        #     clip_text_nodes[1][1]["inputs"]["text"] = negative_prompt
        for node_id, node_data in self._find_nodes_by_type(workflow, "DeepTranslatorTextNode"):
            node_data["inputs"]["text"] = prompt

        for node_id, node_data in self._find_nodes_by_type(
            workflow, "EmptyLatentImage"
        ):
            inputs = node_data.get("inputs", {})
            inputs["width"] = width
            inputs["height"] = height
            node_data["inputs"] = inputs

        for node_id, node_data in self._find_nodes_by_type(
            workflow, "EmptySD3LatentImage"
        ):
            inputs = node_data.get("inputs", {})
            inputs["width"] = width
            inputs["height"] = height
            node_data["inputs"] = inputs

        for node_id, node_data in self._find_nodes_by_type(
            workflow, "ModelSamplingFlux"
        ):
            inputs = node_data.get("inputs", {})
            inputs["width"] = width
            inputs["height"] = height
            node_data["inputs"] = inputs

        for node_id, node_data in self._find_nodes_by_type(
            workflow, "CheckpointLoaderSimple"
        ):
            node_data["inputs"]["ckpt_name"] = checkpoint

        for node_id, node_data in self._find_nodes_by_type(workflow, "UNETLoader"):
            node_data["inputs"]["unet_name"] = checkpoint

        return workflow

    def _inject_loras_to_workflow(
        self,
        workflow: Dict[str, Any],
        loras: list[LoraInfo],
    ) -> Dict[str, Any]:
        if not loras:
            return workflow

        workflow = workflow.copy()

        unet_nodes = self._find_nodes_by_type(workflow, "UNETLoader")
        checkpoint_nodes = self._find_nodes_by_type(workflow, "CheckpointLoaderSimple")

        if unet_nodes:
            model_source_node_id, model_source_data = unet_nodes[0]
            current_model_link = [model_source_node_id, 0]
        elif checkpoint_nodes:
            model_source_node_id, model_source_data = checkpoint_nodes[0]
            current_model_link = [model_source_node_id, 0]
        else:
            logger.warning("No UNETLoader or CheckpointLoaderSimple found for LoRA")
            return workflow

        clip_nodes = self._find_nodes_by_type(workflow, "DualCLIPLoader")
        if clip_nodes:
            current_clip_link = [clip_nodes[0][0], 0]
        elif checkpoint_nodes:
            current_clip_link = [checkpoint_nodes[0][0], 1]
        else:
            logger.warning(
                "No DualCLIPLoader or CheckpointLoaderSimple found for LoRA CLIP"
            )
            current_clip_link = [model_source_node_id, 1]

        max_node_id = max(int(nid) for nid in workflow.keys() if nid.isdigit())

        for idx, lora in enumerate(loras):
            lora_node_id = str(max_node_id + idx + 1)
            workflow[lora_node_id] = {
                "inputs": {
                    "lora_name": lora.name,
                    "strength_model": lora.strength,
                    "strength_clip": lora.strength,
                    "model": current_model_link,
                    "clip": current_clip_link,
                },
                "class_type": "LoraLoader",
            }
            current_model_link = [lora_node_id, 0]
            current_clip_link = [lora_node_id, 1]

        model_target_nodes = [
            "SamplerCustomAdvanced",
            "KSampler",
            "ModelSamplingFlux",
            "BasicGuider",
            "BasicScheduler",
        ]

        for target_type in model_target_nodes:
            for node_id, node_data in self._find_nodes_by_type(workflow, target_type):
                inputs = node_data.get("inputs", {})
                for key, value in list(inputs.items()):
                    if (
                        isinstance(value, list)
                        and len(value) == 2
                        and str(value[0]) == str(model_source_node_id)
                    ):
                        if "model" in key.lower():
                            inputs[key] = current_model_link

        return workflow

    def _inject_init_image_to_workflow(
        self,
        workflow: Dict[str, Any],
        image_filename: str,
        strength: float,
    ) -> Dict[str, Any]:
        workflow = workflow.copy()

        max_node_id = max(int(nid) for nid in workflow.keys() if nid.isdigit())
        load_image_node_id = str(max_node_id + 1)

        workflow[load_image_node_id] = {
            "inputs": {
                "image": image_filename,
            },
            "class_type": "LoadImage",
        }

        for node_id, node_data in self._find_nodes_by_type(workflow, "KSampler"):
            node_data["inputs"]["denoise"] = strength

        for node_id, node_data in self._find_nodes_by_type(workflow, "BasicScheduler"):
            node_data["inputs"]["denoise"] = strength

        return workflow

    async def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        async with await self._get_client() as client:
            payload = {"prompt": workflow, "client_id": self.client_id}
            response = await client.post("/prompt", json=payload)
            response.raise_for_status()
            result = response.json()
            return result["prompt_id"]

    async def wait_for_completion(self, prompt_id: str) -> bool:
        async with await self._get_client() as client:
            for _ in range(self.timeout):
                response = await client.get(f"/history/{prompt_id}")
                response.raise_for_status()
                history = response.json()
                if prompt_id in history:
                    return True
                await asyncio.sleep(1)
            return False

    async def get_output_image(self, prompt_id: str) -> Optional[bytes]:
        async with await self._get_client() as client:
            response = await client.get(f"/history/{prompt_id}")
            response.raise_for_status()
            history = response.json()

            if prompt_id not in history:
                return None

            outputs = history[prompt_id].get("outputs", {})
            for node_id, output in outputs.items():
                if "images" in output:
                    for image_info in output["images"]:
                        filename = image_info.get("filename")
                        subfolder = image_info.get("subfolder", "")
                        img_type = image_info.get("type", "output")

                        params = {
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": img_type,
                        }
                        img_response = await client.get("/view", params=params)
                        if img_response.status_code == 200:
                            return img_response.content
            return None

    def _should_skip_injection(self, request: TextToImageRequest) -> bool:
        default_values = {
            "prompt": "",
            "negative_prompt": "",
            "width": 512,
            "height": 512,
            "steps": 20,
            "cfg_scale": 7.5,
        }

        for field, default in default_values.items():
            value = getattr(request, field, None)
            if value is not None and value != default:
                return False

        if request.seed is not None:
            return False
        if request.sampler_name is not None:
            return False
        if request.scheduler is not None:
            return False

        return True

    async def generate_text_to_image(
        self, request: TextToImageRequest
    ) -> Optional[bytes]:
        seed = (
            request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        )

        if self._should_skip_injection(request):
            logger.info("Using pure workflow mode - no parameter injection")
            workflow = request.workflow.copy()
        else:
            workflow = self._inject_parameters_to_workflow(
                workflow=request.workflow,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or "",
                width=request.width,
                height=request.height,
                steps=request.steps,
                seed=seed,
                cfg_scale=request.cfg_scale,
                sampler_name=request.sampler_name,
                scheduler=request.scheduler,
                checkpoint=request.checkpoint,
            )

        if request.loras:
            workflow = self._inject_loras_to_workflow(workflow, request.loras)

        prompt_id = await self.queue_prompt(workflow)
        completed = await self.wait_for_completion(prompt_id)

        if not completed:
            logger.error(f"Prompt {prompt_id} timed out")
            return None

        return await self.get_output_image(prompt_id)

    async def generate_image_to_image(
        self,
        request: ImageToImageRequest,
        init_image: Image.Image,
    ) -> Optional[bytes]:
        seed = (
            request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        )

        image_filename = f"init_{seed}.png"
        uploaded_filename = await self.upload_image(init_image, image_filename)

        workflow = self._inject_parameters_to_workflow(
            workflow=request.workflow,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            width=request.width,
            height=request.height,
            steps=request.steps,
            seed=seed,
            cfg_scale=request.cfg_scale,
            sampler_name=request.sampler_name,
            scheduler=request.scheduler,
            checkpoint=request.checkpoint,
        )

        workflow = self._inject_init_image_to_workflow(
            workflow, uploaded_filename, request.strength
        )

        if request.loras:
            workflow = self._inject_loras_to_workflow(workflow, request.loras)

        prompt_id = await self.queue_prompt(workflow)
        completed = await self.wait_for_completion(prompt_id)

        if not completed:
            logger.error(f"Prompt {prompt_id} timed out")
            return None

        return await self.get_output_image(prompt_id)

    def is_connected(self) -> bool:
        import httpx

        try:
            response = httpx.get(f"{self.base_url}/system_stats", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
