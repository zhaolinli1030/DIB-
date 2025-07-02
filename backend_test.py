import requests
import json

BASE_URL = "http://localhost:8000/api"  # 您的后端API基础URL
FILE_PATH = "uploads/minidata.xlsx"  # 测试文件路径 (需要实际文件存在)
TEST_QUERY = "中国的第二产业发展情况"

def print_response(response: requests.Response, step_name: str):
    """打印响应信息"""
    print(f"--- {step_name} ---")
    print(f"URL: {response.url}")
    print(f"Status Code: {response.status_code}")
    try:
        response_json = response.json()
        print("Response JSON:")
        print(json.dumps(response_json, indent=2, ensure_ascii=False))
        return response_json
    except requests.exceptions.JSONDecodeError:
        print("Response Text (Not JSON):")
        print(response.text)
        return None
    finally:
        print("\\n")

def main():
    session_id = None
    
    # --- 步骤 1: 文件上传 ---
    print(">>> 步骤 1: 文件上传...")
    try:
        with open(FILE_PATH, "rb") as f:
            files = {"file": (FILE_PATH.split('/')[-1], f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
            # 注意：如果后端期望 session_id 在 Form 中，即使是初次上传，也可能需要传递
            # 如果初次上传不需要 session_id，可以将 data={"session_id": ""} 或不传 data
            response_upload = requests.post(f"{BASE_URL}/data/upload", files=files, data={"session_id": ""}) 
        
        upload_result = print_response(response_upload, "文件上传响应")
        
        if upload_result and response_upload.status_code == 200:
            session_id = upload_result.get("session_id")
            print(f"获取到 Session ID: {session_id}\\n")
        else:
            print("文件上传失败或未能获取 session_id，测试中止。")
            if upload_result and "detail" in upload_result:
                 print(f"错误详情: {upload_result['detail']}")
            return
            
    except FileNotFoundError:
        print(f"错误: 测试文件 {FILE_PATH} 未找到。请确保文件路径正确，并从 datainsightbot/ 目录运行脚本。")
        return
    except requests.exceptions.ConnectionError as e:
        print(f"错误: 无法连接到后端服务 {BASE_URL}。请确保后端服务正在运行。 {e}")
        return
    except Exception as e:
        print(f"步骤 1 文件上传时发生意外错误: {e}")
        return

    if not session_id:
        print("未能从上传响应中获取 session_id，测试无法继续。")
        return

    # --- 步骤 2: 获取数据预览 ---
    print(f">>> 步骤 2: 获取数据预览 (Session ID: {session_id})...")
    try:
        response_preview = requests.get(f"{BASE_URL}/data/{session_id}/preview?rows=5")
        print_response(response_preview, "数据预览响应")
    except requests.exceptions.ConnectionError as e:
        print(f"错误: 无法连接到后端服务 {BASE_URL}。 {e}")
        return
    except Exception as e:
        print(f"步骤 2 获取数据预览时发生意外错误: {e}")
        return

    # --- 步骤 3: 提交分析查询 ---
    print(f">>> 步骤 3: 提交分析查询 (Session ID: {session_id})...")
    try:
        query_payload = {"query": TEST_QUERY}
        # 假设分析查询接口需要 Content-Type: application/json
        headers = {"Content-Type": "application/json"}
        response_query = requests.post(f"{BASE_URL}/analysis/{session_id}/query", data=json.dumps(query_payload), headers=headers)
        print_response(response_query, "分析查询响应")
    except requests.exceptions.ConnectionError as e:
        print(f"错误: 无法连接到后端服务 {BASE_URL}。 {e}")
        return
    except Exception as e:
        print(f"步骤 3 提交分析查询时发生意外错误: {e}")
        return

if __name__ == "__main__":
    main() 