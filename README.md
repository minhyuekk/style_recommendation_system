# 의상 및 스타일 추천 시스템👗 
#### ※개인 프로젝트
<br>

## 개요
### 이미지 속 인물의 의상을 분석하여 스타일을 추출하고, 사용자의 체형에 맞게 스타일을 추천해 주는 시스템
<br>

## 전체 기능
![image](https://github.com/user-attachments/assets/65678cfe-2dea-4e80-81f7-2c5ffe7a1306)

 사용자가 패션 이미지를 입력하면 어떤 의상을 입고 있는지, 어떤 속성을 가진 의상인지 종합하여 스타일 결과를 출력<br>
 그리고, 사용자의 키와 몸무게 정보를 바탕으로 체형을 분석하고, 입력한 이미지 스타일과의 매칭 및 추천
 
 <br>

 ## 사용된 기술
 ### YOLOv8
 YOLOv8을 사용한 의상 객체 검출

 ### ResNet50
 ResNet50을 사용한 의상 속성 분류

 ### OpenAI CLIP
 이미지와 의상 정보를 매칭하여 스타일 출력

 <br>

 ## 데이터셋
 ### [DeepFashion2 Datasets](https://github.com/switchablenorms/DeepFashion2)
 의상 객체 검출을 위해 사용됨

 ### [Fashionpedia Datasets](https://fashionpedia.github.io/home/Fashionpedia_download.html)
 의상 속성 분류를 위해 사용됨
 
 <br>

 ## 기능 설명
 DeepFasion2 데이터셋을 사용하여 YOLOv8s 모델 학습<br>
 ![image](https://github.com/user-attachments/assets/eb2176a5-38cb-45f8-a277-487755322fe1)
 <br>
 (의상 객체 검출 결과)

 <br>

 Fasionpedia 데이터셋을 사용하여 ResNet50 모델 학습<br>
 ![image](https://github.com/user-attachments/assets/dbb7ccc6-08ec-42b5-9e9e-6ffbdbc38323)
 <br>
 (의상 속성 분류 결과)

 <br>

 ResNet50모델 학습 결과<br>
 ![image](https://github.com/user-attachments/assets/116bdc57-d0de-4807-bf69-1fa97ed01d36)
 <br>
 (Accuracy: 0.977, Loss: 0.061)

 <br>

 OpenAI CLIP으로 입력된 이미지와 텍스트 템플릿 매칭을 통해 유사도 계산을 진행 -> 최종 스타일 출력<br>
 ![image](https://github.com/user-attachments/assets/b65e965c-b567-4c93-9565-7d749cf0903c)
 <br>

 <br>

 ## 수행 결과
 ![image](https://github.com/user-attachments/assets/054116d5-3756-4c11-b381-2ead265e159b)

 <br>

 ## 참고 문헌
   https://github.com/switchablenorms/DeepFashion2
   <br>
   https://fashionpedia.github.io/home/Fashionpedia_download.html
   <br>
   https://huggingface.co/Bingsu/adetailer/blob/main/deepfashion2_yolov8s-seg.pt
   <br>
   https://github.com/openai/CLIP
   <br>
   https://paperswithcode.com/dataset/deepfashion2
