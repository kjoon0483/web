# =====================================================
# 손글씨 숫자 인식 프로그램
# MNIST 데이터셋으로 학습한 신경망으로 손글씨 숫자를 인식합니다.
# =====================================================

import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw
import pickle
import os
import threading

# 머신러닝 라이브러리
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# 저장된 모델 파일 경로
모델_파일 = "손글씨_모델.pkl"
스케일러_파일 = "손글씨_스케일러.pkl"


def 모델_학습(진행상황_콜백=None):
    """MNIST 데이터셋으로 신경망 모델을 학습합니다."""

    if 진행상황_콜백:
        진행상황_콜백("📥 MNIST 데이터를 다운로드하는 중... (처음 1회만, 약 1분 소요)")

    # MNIST 데이터셋 불러오기 (28x28 픽셀 이미지 → 784개 특성)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)

    # 학습 속도를 위해 앞 20,000장만 사용 (정확도 ~97% 유지)
    학습수 = 20000
    X_학습 = X[:학습수]
    y_학습 = y[:학습수]

    # 테스트는 60000번 이후 데이터 사용
    X_테스트 = X[60000:]
    y_테스트 = y[60000:]

    if 진행상황_콜백:
        진행상황_콜백("⚙️ 데이터 정규화 중...")

    # 픽셀값(0~255)을 평균 0, 표준편차 1로 정규화
    스케일러 = StandardScaler()
    X_학습 = 스케일러.fit_transform(X_학습)
    X_테스트 = 스케일러.transform(X_테스트)

    if 진행상황_콜백:
        진행상황_콜백("🧠 신경망 학습 중... (30초~1분 소요, 잠시만 기다려주세요)")

    # 다층 퍼셉트론(MLP): 입력784 → 은닉128 → 출력10
    모델 = MLPClassifier(
        hidden_layer_sizes=(128,),     # 은닉층 1개 (빠른 학습)
        activation='relu',             # 활성화 함수
        max_iter=20,                   # 최대 반복 횟수
        random_state=42,              # 재현성을 위한 난수 시드
        learning_rate_init=0.001,     # 초기 학습률
        early_stopping=True,          # 조기 종료 (과적합 방지)
        validation_fraction=0.1,      # 검증 데이터 비율
        n_iter_no_change=5,           # 개선 없을 때 종료 기준
        verbose=False
    )
    모델.fit(X_학습, y_학습)

    # 테스트 데이터로 정확도 평가
    정확도 = 모델.score(X_테스트, y_테스트)

    if 진행상황_콜백:
        진행상황_콜백(f"💾 모델 저장 중... (정확도: {정확도*100:.1f}%)")

    # 모델과 스케일러를 파일로 저장 (다음 실행 시 재학습 불필요)
    with open(모델_파일, 'wb') as f:
        pickle.dump(모델, f)
    with open(스케일러_파일, 'wb') as f:
        pickle.dump(스케일러, f)

    return 모델, 스케일러, 정확도


def 모델_불러오기():
    """저장된 모델 파일이 있으면 불러옵니다."""
    if os.path.exists(모델_파일) and os.path.exists(스케일러_파일):
        with open(모델_파일, 'rb') as f:
            모델 = pickle.load(f)
        with open(스케일러_파일, 'rb') as f:
            스케일러 = pickle.load(f)
        return 모델, 스케일러
    return None, None


# =====================================================
# GUI 애플리케이션 클래스
# =====================================================
class 손글씨인식앱:
    def __init__(self, root):
        self.root = root
        self.root.title("✍️ 손글씨 숫자 인식기")
        self.root.resizable(False, False)
        self.root.configure(bg='#f0f0f0')

        self.모델 = None
        self.스케일러 = None
        self.모델_준비됨 = False

        # 캔버스 크기 (MNIST는 28x28이지만 그리기 편의를 위해 280x280 사용)
        self.캔버스_크기 = 280
        self.붓_크기 = 14  # 붓 굵기

        # 그리기 상태 변수
        self.그리기중 = False
        self.이전_x = None
        self.이전_y = None

        # PIL 이미지 (픽셀 데이터 저장용, 검은 배경)
        self.이미지 = Image.new('L', (self.캔버스_크기, self.캔버스_크기), 0)
        self.그리기도구 = ImageDraw.Draw(self.이미지)

        self._화면_구성()
        self._이벤트_연결()

        # 앱 시작 시 모델 불러오기 또는 학습 (백그라운드 스레드에서 실행)
        threading.Thread(target=self._모델_초기화, daemon=True).start()

    def _화면_구성(self):
        """UI 화면을 구성합니다."""

        # 상단 제목
        제목_프레임 = tk.Frame(self.root, bg='#2c3e50', pady=10)
        제목_프레임.pack(fill=tk.X)
        tk.Label(제목_프레임, text="손글씨 숫자 인식기",
                 font=('맑은 고딕', 18, 'bold'),
                 fg='white', bg='#2c3e50').pack()
        tk.Label(제목_프레임, text="아래 캔버스에 숫자(0~9)를 마우스로 그려주세요",
                 font=('맑은 고딕', 10),
                 fg='#bdc3c7', bg='#2c3e50').pack()

        # 중앙 영역 (캔버스 + 결과)
        중앙_프레임 = tk.Frame(self.root, bg='#f0f0f0', padx=20, pady=15)
        중앙_프레임.pack()

        # 그림 캔버스 (검은 배경, 흰색으로 그리기)
        캔버스_프레임 = tk.Frame(중앙_프레임, bd=3, relief=tk.SUNKEN)
        캔버스_프레임.pack(side=tk.LEFT, padx=(0, 20))
        self.캔버스 = tk.Canvas(캔버스_프레임,
                                width=self.캔버스_크기,
                                height=self.캔버스_크기,
                                bg='black', cursor='crosshair')
        self.캔버스.pack()

        # 오른쪽 결과 패널
        결과_패널 = tk.Frame(중앙_프레임, bg='#f0f0f0', width=180)
        결과_패널.pack(side=tk.LEFT, fill=tk.Y)

        # 인식 결과 숫자 표시
        tk.Label(결과_패널, text="인식 결과",
                 font=('맑은 고딕', 12, 'bold'),
                 bg='#f0f0f0', fg='#2c3e50').pack(pady=(0, 5))

        결과_박스 = tk.Frame(결과_패널, bg='white', bd=2, relief=tk.GROOVE,
                             width=150, height=120)
        결과_박스.pack()
        결과_박스.pack_propagate(False)

        self.결과_레이블 = tk.Label(결과_박스, text="?",
                                    font=('맑은 고딕', 60, 'bold'),
                                    fg='#e74c3c', bg='white')
        self.결과_레이블.pack(expand=True)

        # 확신도 표시
        self.확신도_레이블 = tk.Label(결과_패널, text="확신도: -",
                                      font=('맑은 고딕', 11),
                                      bg='#f0f0f0', fg='#27ae60')
        self.확신도_레이블.pack(pady=5)

        # 확률 진행 바
        self.확률_바 = ttk.Progressbar(결과_패널, length=150,
                                        mode='determinate', maximum=100)
        self.확률_바.pack()

        # 상위 후보 목록
        tk.Label(결과_패널, text="\n상위 3개 후보",
                 font=('맑은 고딕', 10, 'bold'),
                 bg='#f0f0f0', fg='#2c3e50').pack()

        self.후보_레이블 = tk.Label(결과_패널, text="",
                                    font=('맑은 고딕', 10),
                                    bg='#f0f0f0', fg='#7f8c8d',
                                    justify=tk.LEFT)
        self.후보_레이블.pack()

        # 버튼 영역
        버튼_프레임 = tk.Frame(self.root, bg='#f0f0f0', pady=10)
        버튼_프레임.pack()

        # 인식하기 버튼 (모델 준비 전에는 비활성화)
        self.인식_버튼 = tk.Button(버튼_프레임, text="⏳ 모델 준비 중...",
                                   command=self.숫자_인식,
                                   font=('맑은 고딕', 12, 'bold'),
                                   bg='#95a5a6', fg='white',
                                   padx=20, pady=5,
                                   relief=tk.FLAT, cursor='watch',
                                   state=tk.DISABLED)
        self.인식_버튼.pack(side=tk.LEFT, padx=8)

        tk.Button(버튼_프레임, text="🗑 지우기",
                  command=self.캔버스_초기화,
                  font=('맑은 고딕', 12, 'bold'),
                  bg='#e74c3c', fg='white',
                  padx=20, pady=5,
                  relief=tk.FLAT, cursor='hand2').pack(side=tk.LEFT, padx=8)

        # 하단 로딩 진행 바 (학습 중에만 표시)
        self.로딩_바 = ttk.Progressbar(self.root, mode='indeterminate', length=500)
        self.로딩_바.pack(fill=tk.X, padx=20, pady=(0, 5))
        self.로딩_바.start(10)  # 애니메이션 시작

        # 하단 상태 표시줄
        self.상태_레이블 = tk.Label(self.root,
                                    text="⏳ 모델을 불러오는 중... 잠시만 기다려주세요.",
                                    font=('맑은 고딕', 9),
                                    bg='#ecf0f1', fg='#e67e22',
                                    anchor=tk.W, padx=10, pady=4)
        self.상태_레이블.pack(fill=tk.X, side=tk.BOTTOM)

    def _이벤트_연결(self):
        """마우스 이벤트를 연결합니다."""
        self.캔버스.bind('<Button-1>', self._그리기_시작)
        self.캔버스.bind('<B1-Motion>', self._그리기)
        self.캔버스.bind('<ButtonRelease-1>', self._그리기_종료)
        # 더블클릭하면 자동 인식
        self.캔버스.bind('<Double-Button-1>', lambda e: self.숫자_인식())

    def _모델_초기화(self):
        """모델을 불러오거나 새로 학습합니다 (백그라운드 실행)."""

        def 상태_업데이트(메시지):
            """UI 스레드에서 안전하게 상태 레이블을 업데이트합니다."""
            self.root.after(0, lambda: self.상태_레이블.config(text=메시지))

        # 저장된 모델 먼저 탐색
        모델, 스케일러 = 모델_불러오기()

        if 모델 is not None:
            self.모델 = 모델
            self.스케일러 = 스케일러
            self.root.after(0, self._모델_준비완료, "✅ 저장된 모델 불러오기 완료! 숫자를 그려보세요.")
        else:
            # 저장된 모델 없으면 새로 학습
            try:
                모델, 스케일러, 정확도 = 모델_학습(진행상황_콜백=상태_업데이트)
                self.모델 = 모델
                self.스케일러 = 스케일러
                완료_메시지 = f"✅ 학습 완료! 정확도: {정확도*100:.1f}% | 숫자를 그려보세요."
                self.root.after(0, self._모델_준비완료, 완료_메시지)
            except Exception as e:
                오류_메시지 = str(e)
                self.root.after(0, lambda msg=오류_메시지: self.상태_레이블.config(
                    text=f"❌ 오류 발생: {msg}", fg='#e74c3c'))

    def _모델_준비완료(self, 메시지):
        """모델 준비가 완료되면 버튼을 활성화합니다."""
        self.모델_준비됨 = True

        # 인식 버튼 활성화
        self.인식_버튼.config(
            text="✅ 인식하기",
            bg='#27ae60',
            cursor='hand2',
            state=tk.NORMAL
        )

        # 로딩 바 숨기기
        self.로딩_바.stop()
        self.로딩_바.pack_forget()

        # 상태 메시지 업데이트
        self.상태_레이블.config(text=메시지, fg='#27ae60')

    def _그리기_시작(self, event):
        """마우스 클릭 시 그리기를 시작합니다."""
        self.그리기중 = True
        self.이전_x = event.x
        self.이전_y = event.y

    def _그리기(self, event):
        """마우스 드래그로 캔버스에 선을 그립니다."""
        if not self.그리기중:
            return

        x, y = event.x, event.y
        r = self.붓_크기

        # tkinter 캔버스에 흰색 원 그리기 (화면 표시용)
        self.캔버스.create_oval(x - r, y - r, x + r, y + r,
                                fill='white', outline='white')
        if self.이전_x is not None:
            self.캔버스.create_line(self.이전_x, self.이전_y, x, y,
                                    fill='white', width=r * 2,
                                    capstyle=tk.ROUND, joinstyle=tk.ROUND)

        # PIL 이미지에도 동일하게 그리기 (모델 입력용)
        self.그리기도구.ellipse([x - r, y - r, x + r, y + r], fill=255)
        if self.이전_x is not None:
            self.그리기도구.line([self.이전_x, self.이전_y, x, y],
                                fill=255, width=r * 2)

        self.이전_x = x
        self.이전_y = event.y

    def _그리기_종료(self, event):
        """마우스 버튼을 놓으면 그리기를 종료합니다."""
        self.그리기중 = False
        self.이전_x = None
        self.이전_y = None

    def 숫자_인식(self):
        """캔버스에 그린 숫자를 인식합니다."""
        if not self.모델_준비됨:
            return  # 버튼이 비활성화되어 있어서 여기까지 오지 않지만 방어 코드

        # 이미지 전처리: 280x280 → 28x28 축소 후 1차원 배열로 변환
        이미지_28 = self.이미지.resize((28, 28), Image.LANCZOS)
        픽셀_배열 = np.array(이미지_28, dtype=np.float64).reshape(1, -1)

        # 학습 때와 동일한 방식으로 정규화
        픽셀_정규화 = self.스케일러.transform(픽셀_배열)

        # 예측 실행
        예측_숫자 = self.모델.predict(픽셀_정규화)[0]
        확률_배열 = self.모델.predict_proba(픽셀_정규화)[0]
        최대_확률 = float(max(확률_배열)) * 100

        # 결과 화면 업데이트
        self.결과_레이블.config(text=str(예측_숫자))
        self.확신도_레이블.config(text=f"확신도: {최대_확률:.1f}%")
        self.확률_바['value'] = 최대_확률

        # 상위 3개 후보 표시
        상위3 = sorted(enumerate(확률_배열), key=lambda x: x[1], reverse=True)[:3]
        후보_텍스트 = "\n".join([f"  숫자 {숫자}: {확률*100:.1f}%"
                                 for 숫자, 확률 in 상위3])
        self.후보_레이블.config(text=후보_텍스트)

        self.상태_레이블.config(
            text=f"✅ 인식 완료: {예측_숫자}번 (확신도 {최대_확률:.1f}%)",
            fg='#2980b9')

    def 캔버스_초기화(self):
        """캔버스와 PIL 이미지를 모두 초기화합니다."""
        self.캔버스.delete("all")
        self.이미지 = Image.new('L', (self.캔버스_크기, self.캔버스_크기), 0)
        self.그리기도구 = ImageDraw.Draw(self.이미지)
        self.결과_레이블.config(text="?")
        self.확신도_레이블.config(text="확신도: -")
        self.확률_바['value'] = 0
        self.후보_레이블.config(text="")
        if self.모델_준비됨:
            self.상태_레이블.config(text="🗑 캔버스를 지웠습니다. 새 숫자를 그려보세요.",
                                    fg='#7f8c8d')


# =====================================================
# 프로그램 진입점
# =====================================================
if __name__ == "__main__":
    root = tk.Tk()
    앱 = 손글씨인식앱(root)
    root.mainloop()
