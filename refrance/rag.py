import time
import shutil
import os
import pymongo
import datetime

from typing import Union
from bson import ObjectId
from fastapi import APIRouter, UploadFile, Response, status
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel
from qwen_vl_utils import process_vision_info
from typing import Annotated
from rag.database import MoyuClient
from rag.text import TextProcessor
from api.dataclass import MessageList, ChatRequest


router = APIRouter(prefix="/rag")

# load model
LLM = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="cuda:1"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

rerank_model = AutoModel.from_pretrained('jinaai/jina-reranker-m0',
                                                torch_dtype="auto",
                                                trust_remote_code=True,
                                                device_map="cuda:0"
                                            )

# load database and text processor
client = MoyuClient(r"./data/db/milvus_demo.db")
text_processor = TextProcessor()



# text operations
@router.post("/add_text")
async def add_text(file: UploadFile, collection_name: str = "rag_embeddings"):
    """
    Add a text file to the database
    """
    
    content = file.file.read().decode('utf-8')

    client.init_collection(collection_name)

    #save file
    if not os.path.exists(f"./data/text/{collection_name}"):
        os.makedirs(f"./data/text/{collection_name}")
    save_path = f"./data/text/{collection_name}/{int(time.time())}.{file.filename.split('.')[-1] if file.filename else '.tmp'}"
    with open(save_path, "w+") as f:
        f.write(content)

    # chunk text and embedding
    chunks = text_processor.split_markdown_for_rag(content, min_words=10, include_subsections=False)
    vectors = client.get_text_embeddings(texts=chunks, is_query=False)
    # print dim of vector
    print("Dim:", client.embedding_model.config.hidden_size, vectors[0].shape)

    # insert data
    data = [
        {"embedding": vectors[i], "type":"text", "text": chunks[i], "path":os.path.abspath(save_path), "subject": "capp"}
        for i in range(len(vectors))
    ]

    print("Data has", len(data), "entities, each with fields: ", data[0].keys())

    # insert data
    res = client.insert(data=data, collection_name=collection_name)
    return {
        "status": "success",
        "message": f"The {file.filename} has {len(data)} entities inserted into collection {collection_name}",
        }

@router.get("/query")
async def query(text: str, collection_name="rag_embeddings", limit: int=10):
    """
    Query the database with text
    """

    query_vectors = client.get_text_embeddings(texts=[text]).squeeze().tolist()
    res = client.search(
        data=[query_vectors],  # query vectors
        collection_name=collection_name,
        limit=limit,  # number of returned entities
        output_fields=["text", "subject", "path"],  # specifies fields to be returned
    )

    return [item["entity"] for item in res[0]]


# collection operations
@router.post("/create_collection")
async def create_collection(collection_name: str):
    """
    Create a collection in the database
    """
    client.init_collection(collection_name)

    if not os.path.exists(f"./data/text/{collection_name}"):
        os.makedirs(f"./data/text/{collection_name}")

    return {
        "status": "success",
        "message": f"Collection {collection_name} created"
        }

@router.get("/list_collections")
async def list_collections():
    """
    List all collections in the database
    """
    return client.list_collections()

@router.delete("/delete_collection")
async def delete_collection(collection_name: str):
    """
    Delete a collection from the database
    """
    client.drop_collection(collection_name)
    shutil.rmtree(f"./data/text/{collection_name}")
    return {
        "status": "success", 
        "message": f"Collection {collection_name} deleted"
        }

# chat operations
@router.post("/chat")
async def chat(history: MessageList, collection_name: str = "rag_embeddings", retrieval:bool=True, limit: int = 4):
    messages = history.model_dump()["messages"]
    # 检索
    if retrieval:
        last_message = messages.pop()
        if last_message["role"] != "user":
            return {
                "status": "failed",
                "message": "Wrong role"
            }
        texts = []
        images = []
        for content in last_message["content"]:
            if content["type"] == "text":
                texts.append(content["text"])
            elif content["type"] == "image":
                images.append(f"file://{os.path.abspath(content['image'])}")
        
        # query database
        query_vector = client.get_fused_embeddings(texts=texts, images=images).squeeze().tolist()
        query_res = client.search(
            data=[query_vector],
            collection_name=collection_name,
            limit=limit,
            output_fields=["text", "subject", "path", "type"],
        )

        # rerank
        rerank_num = limit
        for i in range(len(query_res[0])):
            entity = query_res[0][i]['entity']
            if entity['type'] == "text":
                scores = rerank_model.compute_score([[texts[-1], entity['text']]], max_length=1024, doc_type="text")
                query_res[0][i]['score'] = scores
            elif entity['type'] == "image":
                scores = rerank_model.compute_score([[texts[-1], entity['path']]], max_length=1024, doc_type="image")
                query_res[0][i]['score'] = scores

        query_res[0].sort(key=lambda x: x['score'], reverse=True)

        # generate message
        query_content_list = []

        for i in range(rerank_num):
            entity = query_res[0][i]['entity']
            if entity['type'] == "text":
                query_content_list.append(
                    {"type": "text", "text": f"{i+1}.{entity['text']}(来源为{entity['path']})\n"}
                    )
            elif entity['type'] == "image":
                query_content_list.append(
                    {"type": "text", "text": f"{i+1}.{entity['text']}"}
                    )
                query_content_list.append(
                    {"type": "image", "image": f"file://{os.path.abspath(entity['path'])}"}
                    )
                query_content_list.append(
                    {"type": "text", "text": f"(来源为{entity['path']})\n"}
                    )
        
        messages += [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "知识库中搜索有如下结果\n"}
                ] + 
                query_content_list + 
                [
                    {"type": "text", "text": f"用户的问题是:{texts[-1]}, 请你根据知识库结果回答用户的问题，在你回答用户问题时，需要过滤掉搜索中无关的项，并在你的回答中涉及知识库内容的地方使用链接形式，根据结果前的序号及结果后的来源分别对文字和图像进行标注，图片标注示例:![序号](来源)，文字标注示例:[序号](来源)"},
                    {"type": "text", "text": r"对于装配工序请使用以下模板进行回答：\n"},
                    {"type": "text", "text": """
                        {
                            "info":{
                                "processID":"工序号",
                                "processName":"工序名称",
                                "processCharacteristic":"工序特征",
                                "processType":"工序类型枚举:assemble/measure/postprocess",
                                "note":"工序备注"
                            },
                            "steps":{
                                "工步名称":"工步文件路径.json"
                            },
                            "processResources":[
                                {
                                    "resourceName":"工艺资源名称",
                                    "resourceID":"工艺资源号",
                                    "specification":"工艺资源规格",
                                    "resourceType":"工艺资源类型枚举:tool/people",
                                    "model":"工艺资源模型路径.step"
                                }    
                            ],
                            "parts":[
                                {
                                    "partID":"装配件代号",
                                    "partName":"装配件名称",
                                    "partNum":1,
                                    "action":"装配动作",
                                    "model":"装配件模型路径.step"
                                }
                            ]
                        }
                        """
                    },
                    {"type": "text", "text": r"对于装配工步请使用以下模板进行回答：\n"},
                    {"type": "text", "text": """
                        {
                            "info":{
                                "stepID":"工步号",
                                "stepName":"工步名称"
                            },
                            "stepContent":[
                                {
                                    "title":"工步基本内容标题",
                                    "detail":"工步基本内容细节1"
                                },
                                {
                                    "title":"测量工步内容标题",
                                    "detail": "测量工步内容细节",
                                    "target":"测量指标",
                                    "inspection_method":"检验方式",
                                    "measure_method":"测量方法",
                                    "times":1,
                                    "resource_name":"资源名称",
                                    "resource_id":"资源代号",
                                    "resource_specification":"资源规格",
                                    "digital":false
                                }
                            ],
                            "annex":"工步附件路径.*",
                            "animation":"工步装配动画"
                        }
                        """
                    },
                ],
            }
        ]

    print(messages)        
    
    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(LLM.device)

    # Inference: Generation of the output
    generated_ids = LLM.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    with open(f"./data/tmp/{int(time.time())}_output.md", "w+", encoding="utf-8") as f:
        for text in output_text:
            f.write(text)
    
    return {
        "response": output_text,
    }

@router.post("/chat/message")
async def chat_message(request: ChatRequest, response: Response):
    # query history
    mClient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mClient["assemblyPlan"]
    sessions = db["sessions"]
    history = sessions.find_one({"_id": ObjectId(request.session_id)}, {"history":1, "_id": 0})
    if history is None:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {"message": "session not found"}
    
    if len(history["history"]) < 1:
        sessions.update_one({"_id": ObjectId(request.session_id)}, {"$set": {"title": request.message[:min(10, len(request.message))]}})

    messages = history["history"]
    # add new message
    if request.image:
        user_message = {"_id":messages[-1]["_id"] + 1 if len(messages) > 0 else 1,"role": "user", "createTime":request.createTime, "content": [
            {"type":"text", "text": request.message}, 
            {"type":"image", "image": request.image}
            ]}
    else:
        user_message = {"_id":messages[-1]["_id"] + 1 if len(messages) > 0 else 1,"role": "user", "createTime": request.createTime, "content": [{"type":"text", "text": request.message}]}
    messages.append(user_message)

    llm_message = {
        "_id": messages[-1]["_id"] + 1 ,
        "role": "assistant"
    }

    # get abs apth
    for i in range(len(messages)):
        for j in range(len(messages[i]["content"])):
            if messages[i]["content"][j]["type"] == "image":
                image_name = messages[i]["content"][j]["image"]
                image_path = f"./data/tmp/img/{image_name}.png"
                messages[i]["content"][j]["image"] = f"file://{os.path.abspath(image_path)}"

    relatedDocs = []
    # 检索
    if request.collection_name:
        last_message = messages.pop()
        if last_message["role"] != "user":
            response.status_code = status.HTTP_406_NOT_ACCEPTABLE
            return {
                "status": "failed",
                "message": "Wrong role"
            }
        texts = []
        images = []
        for content in last_message["content"]:
            if content["type"] == "text":
                texts.append(content["text"])
            elif content["type"] == "image":
                images.append(content["image"])
        
        # query database
        query_vector = client.get_fused_embeddings(texts=texts, images=images).squeeze().tolist()
        query_res = client.search(
            data=[query_vector],
            collection_name=request.collection_name,
            limit=request.limit,
            output_fields=["text", "subject", "path", "type"],
        )

        # rerank
        rerank_num = request.limit
        for i in range(len(query_res[0])):
            entity = query_res[0][i]['entity']
            if entity['type'] == "text":
                scores = rerank_model.compute_score([[texts[-1], entity['text']]], max_length=1024, doc_type="text")
                query_res[0][i]['score'] = scores
            elif entity['type'] == "image":
                scores = rerank_model.compute_score([[texts[-1], entity['path']]], max_length=1024, doc_type="image")
                query_res[0][i]['score'] = scores

        query_res[0].sort(key=lambda x: x['score'], reverse=True)

        # generate message
        query_content_list = []
        

        for i in range(rerank_num):
            entity = query_res[0][i]['entity']
            
            if entity['type'] == "text":
                relatedDocs.append({"_id": i, "title":entity["type"], "content": entity['text'], "score": query_res[0][i]['score']})
                query_content_list.append(
                    {"type": "text", "text": f"{i+1}.{entity['text']}(来源为{entity['path']})\n"}
                    )
            elif entity['type'] == "image":
                query_content_list.append(
                    {"type": "text", "text": f"{i+1}.{entity['text']}"}
                    )
                query_content_list.append(
                    {"type": "image", "image": f"file://{os.path.abspath(entity['path'])}"}
                    )
                query_content_list.append(
                    {"type": "text", "text": f"(来源为{entity['path']})\n"}
                    )
        
        # assemble messages, warning field content eq contents
        messages += [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "知识库中搜索有如下结果\n"}
                ] + 
                query_content_list + 
                [
                    {"type": "text", "text": "用户提供了以下图片"}
                ] +
                [
                    {"type": "image", "image": content["image"]} for content in last_message["content"] if content["type"] == "image"
                ] +
                [
                    {"type": "text", "text": f"用户的问题是:{texts[-1]}, 请你根据知识库结果回答用户的问题，在你回答用户问题时，需要过滤掉搜索中无关的项，并在你的回答中涉及知识库内容的地方使用链接形式，根据结果前的序号及结果后的来源分别对文字和图像进行标注，图片标注示例:![序号](来源)，文字标注示例:[序号](来源)"},
                    {"type": "text", "text": r"对于装配工序请使用以下模板进行回答：\n"},
                    {"type": "text", "text": """
                        {
                            "info":{
                                "processID":"工序号",
                                "processName":"工序名称",
                                "processCharacteristic":"工序特征",
                                "processType":"工序类型枚举:assemble/measure/postprocess",
                                "note":"工序备注"
                            },
                            "steps":{
                                "工步名称":"工步文件路径.json"
                            },
                            "processResources":[
                                {
                                    "resourceName":"工艺资源名称",
                                    "resourceID":"工艺资源号",
                                    "specification":"工艺资源规格",
                                    "resourceType":"工艺资源类型枚举:tool/people",
                                    "model":"工艺资源模型路径.step"
                                }    
                            ],
                            "parts":[
                                {
                                    "partID":"装配件代号",
                                    "partName":"装配件名称",
                                    "partNum":1,
                                    "action":"装配动作",
                                    "model":"装配件模型路径.step"
                                }
                            ]
                        }
                        """
                    },
                    {"type": "text", "text": r"对于装配工步请使用以下模板进行回答：\n"},
                    {"type": "text", "text": """
                        {
                            "info":{
                                "stepID":"工步号",
                                "stepName":"工步名称"
                            },
                            "stepContent":[
                                {
                                    "title":"工步基本内容标题",
                                    "detail":"工步基本内容细节1"
                                },
                                {
                                    "title":"测量工步内容标题",
                                    "detail": "测量工步内容细节",
                                    "target":"测量指标",
                                    "inspection_method":"检验方式",
                                    "measure_method":"测量方法",
                                    "times":1,
                                    "resource_name":"资源名称",
                                    "resource_id":"资源代号",
                                    "resource_specification":"资源规格",
                                    "digital":false
                                }
                            ],
                            "annex":"工步附件路径.*",
                            "animation":"工步装配动画"
                        }
                        """
                    },
                ],
            }
        ]

    print(messages)        
    
    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(LLM.device)

    # Inference: Generation of the output
    generated_ids = LLM.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    with open(f"./data/tmp/{int(time.time())}_output.md", "w+", encoding="utf-8") as f:
        for text in output_text:
            f.write(text)

    # message add to db

    # back to img name
    for i in range(len(user_message["content"])):
        if user_message["content"][i]["type"] == "image":
            image_path = user_message["content"][i]["image"]
            image_name = image_path.split("/")[-1].split(".")[0]
            user_message["content"][i]["image"] = image_name

    sessions.update_one(
        {"_id": ObjectId(request.session_id)},
        {"$push": {"history": user_message}}
    )

    llm_message["createTime"] = datetime.datetime.now().isoformat(),
    llm_message["content"]=[
            {
                "type": "text",
                "text": text
            }
            for text in output_text
        ]

    sessions.update_one(
        {"_id": ObjectId(request.session_id)},
        {"$push": {"history": llm_message}}
    )
    
    return {
        "_id": llm_message["_id"],
        "role": 'assistant',
        "content": llm_message["content"],
        "createTime": llm_message["createTime"],
        "relatedDocs": relatedDocs
    }