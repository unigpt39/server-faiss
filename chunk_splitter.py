
# 분석 완료
# expand_more
# manual.txt 파일에는 회사 운영과 관련된 다양한 규정이 포함되어 있습니다. 각 규정은 '제1조', '제2조'와 같은 조항으로 구분되어 있으며, 규정의 제목과 조항은 문서의 의미론적 단위로 볼 수 있습니다. 따라서, 규정의 제목과 조항을 기준으로 텍스트를 분할하는 것이 적합합니다.

# 이러한 특징을 고려할 때, RecursiveCharacterTextSplitter보다는 **정규 표현식(Regular Expression)**을 사용하는 것이 더 적합합니다. 정규 표현식은 특정 패턴을 가진 텍스트를 찾고 추출하는 데 사용되는 강력한 도구입니다.

# 다음과 같은 정규 표현식을 사용하여 규정의 제목과 조항을 기준으로 텍스트를 분할할 수 있습니다.

# 제\s*\d+[조|항]\s*【.*】
# 이 정규 표현식은 '제'로 시작하고 공백이 있거나 없을 수 있으며, 그 뒤에 숫자 하나 이상과 '조' 또는 '항', 그리고 다시 공백이 있거나 없을 수 있으며 '【'와 '】' 사이에 제목이 있는 패턴을 찾습니다.

# 이 정규 표현식을 사용하면 각 조항을 정확하게 분할할 수 있으며, 규정의 제목도 함께 추출할 수 있습니다.

import re

def split_text_by_regulation(text):
  """
  정규 표현식을 사용하여 텍스트를 규정 제목과 조항을 기준으로 분할합니다.

  Args:
      text (str): 분할할 텍스트

  Returns:
      list: 분할된 텍스트 조각의 리스트
  """
  pattern = r'제\s*\d+[조|항]\s*【.*】'  # 규정 제목과 조항 패턴
  chunks = re.split(pattern, text)
  chunks = [chunk.strip() for chunk in chunks if chunk.strip()]  # 빈 문자열 제거
  return chunks

with open("manual.txt", "r", encoding="utf-8") as f:
  text = f.read()

chunks = split_text_by_regulation(text)

for chunk in chunks:
  print(chunk)
  print("=" * 30)  # 조항 구분선
