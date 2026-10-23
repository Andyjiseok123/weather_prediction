import requests  # requests 모듈 임포트
# URL과 저장 경로 변수를 지정합니다.
url = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php?tm1=20160101&tm2=20261021&stn=0&help=1&authKey=qnBabhJnRp2wWm4SZ7adQg'
response = requests.get(url=url)
csv_text = response.content.decode('utf-8')
# 2. 파일로 저장
with open('weather_data_raw.csv', 'w', encoding='cp-949', newline='') as f:
    f.write(csv_text)
