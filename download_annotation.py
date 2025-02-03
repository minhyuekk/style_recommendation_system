import requests

# S3 URL
url = "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json"
output_path = "../feature_extractor/data/labels/instances_attributes_train2020.json"

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    print(f"파일이 {output_path}에 저장되었습니다.")
else:
    print("파일 다운로드 실패:", response.status_code)
