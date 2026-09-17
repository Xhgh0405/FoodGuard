from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph, Spacer, ListFlowable
from reportlab.lib.styles import getSampleStyleSheet

NAVY = HexColor('#1D3557')
BLUE = HexColor('#2E6BFF')
TEAL = HexColor('#168E95')
LIGHT = HexColor('#EEF5FF')
GRAY = HexColor('#58606B')
GREEN = HexColor('#2D8A5D')
ORANGE = HexColor('#E67E22')
RED = HexColor('#C0392B')
WHITE = HexColor('#FFFFFF')

W, H = landscape(letter)

def draw_header(c, title, subtitle=None):
    c.setFillColor(NAVY)
    c.setFont('Helvetica-Bold', 22)
    c.drawString(0.8*inch, H - 0.8*inch, title)
    if subtitle:
        c.setFillColor(GRAY)
        c.setFont('Helvetica', 12)
        c.drawString(0.8*inch, H - 1.2*inch, subtitle)
    c.setStrokeColor(BLUE)
    c.setLineWidth(2)
    c.line(0.6*inch, H - 1.5*inch, W - 0.6*inch, H - 1.5*inch)


def draw_box(c, x, y, w, h, title, fill, text_color=WHITE, font_size=14):
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, 12, fill=1, stroke=0)
    c.setFillColor(text_color)
    c.setFont('Helvetica-Bold', font_size)
    lines = title.split('\n')
    for i, line in enumerate(lines):
        c.drawCentredString(x + w/2, y + h - 20 - i*18, line)


def draw_text_block(c, x, y, w, h, lines, font='Helvetica', size=12, color=NAVY, bold=False):
    c.setFillColor(color)
    c.setFont('Helvetica-Bold' if bold else font, size)
    for idx, line in enumerate(lines):
        c.drawString(x, y + h - idx*20, line)


def draw_bullets(c, x, y, w, lines, font_size=12):
    c.setFillColor(NAVY)
    c.setFont('Helvetica', font_size)
    for i, line in enumerate(lines):
        c.drawString(x + 0.2*inch, y - i*24, '• ' + line)


def add_page1(c):
    c.setFillColor(LIGHT)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont('Helvetica-Bold', 30)
    c.drawString(0.9*inch, H - 1.1*inch, 'FoodGuard')
    c.setFillColor(BLUE)
    c.setFont('Helvetica', 24)
    c.drawString(0.9*inch, H - 1.9*inch, '基於 MCP 與 RAG 的食品標示智慧合規判讀系統')
    c.setFillColor(GRAY)
    c.setFont('Helvetica', 17)
    c.drawString(0.95*inch, H - 2.8*inch, '專題簡報：系統架構、設計理念與系統規格')
    c.drawString(0.95*inch, H - 3.3*inch, '整合 RAG 檢索、MCP 工具協作、規則引擎與 Streamlit UI')
    c.drawString(0.95*inch, H - 3.8*inch, '針對食品標示、營養宣稱、過敏原與法規依據進行判讀')
    draw_box(c, W - 3.6*inch, H - 4.2*inch, 2.8*inch, 2.2*inch, '系統目標\n1. 提升判讀正確性\n2. 建立法規依據\n3. AI + 規則 + RAG\n4. 提供互動式 Demo', BLUE)
    c.setFillColor(GRAY)
    c.setFont('Helvetica', 10)
    c.drawString(0.9*inch, 0.35*inch, 'FoodGuard｜專題 Demo｜2026')


def add_page2(c):
    draw_header(c, '一、研究背景與系統目標', '食品標示合規檢查需兼顧法規依據、數值判斷與可解釋性')
    draw_bullets(c, 0.8*inch, H - 2.2*inch, 5.0*inch, [
        '食品包裝資訊常涉及法規、營養宣稱與過敏原風險，判讀容易受主觀解讀影響。',
        '一般 AI 回答可能缺乏官方依據，導致判斷結果不具可驗證性。',
        '系統必須結合官方法規資料庫、規則引擎與追蹤來源能力。',
        '為了實務應用，需要提供結論、依據與建議，同時保留 UI 可操作性。'
    ], font_size=12)
    draw_bullets(c, 6.5*inch, H - 2.2*inch, 5.0*inch, [
        '支援食品品名、成分、營養標示與營養宣稱輸入。',
        '利用 RAG 檢索 TFDA 官方文件與法規文本。',
        '結合 MCP 工具協作與 Python deterministic rule engine。',
        '輸出判讀結果、法規證據與可下載離線報告。'
    ], font_size=12)


def add_page3(c):
    draw_header(c, '二、系統架構', '以 User Interface、MCP Client、Server Tools 與 RAG 引擎構成分層式架構')
    boxes = [
        ('Streamlit UI', 0.7*inch, H - 3.0*inch, 1.4*inch, 0.8*inch, BLUE),
        ('MCP Client', 2.4*inch, H - 3.0*inch, 1.5*inch, 0.8*inch, TEAL),
        ('MCP Server', 4.3*inch, H - 3.0*inch, 1.4*inch, 0.8*inch, GREEN),
        ('Rule Engine', 6.3*inch, H - 3.0*inch, 1.6*inch, 0.8*inch, ORANGE),
        ('RAG Retriever', 8.4*inch, H - 3.0*inch, 1.8*inch, 0.8*inch, RED),
        ('Official DOCs', 10.7*inch, H - 3.0*inch, 1.3*inch, 0.8*inch, NAVY),
        ('SQLite / Memory', 4.5*inch, H - 5.0*inch, 2.2*inch, 0.7*inch, BLUE),
        ('FAISS Index', 7.4*inch, H - 5.0*inch, 1.9*inch, 0.7*inch, TEAL),
        ('LLM / Summary', 9.8*inch, H - 5.0*inch, 1.7*inch, 0.7*inch, GREEN),
    ]
    for label, x, y, w, h, fill in boxes:
        draw_box(c, x, y, w, h, label, fill, text_color=WHITE, font_size=11)
    c.setStrokeColor(GRAY)
    c.setLineWidth(1.2)
    for x1, y1, x2, y2 in [
        (2.1*inch, H-3.4*inch, 2.4*inch, H-3.4*inch),
        (3.9*inch, H-3.4*inch, 4.3*inch, H-3.4*inch),
        (5.7*inch, H-3.4*inch, 6.3*inch, H-3.4*inch),
        (7.9*inch, H-3.4*inch, 8.4*inch, H-3.4*inch),
        (10.2*inch, H-3.4*inch, 10.7*inch, H-3.4*inch),
        (5.6*inch, H-3.8*inch, 5.6*inch, H-5.0*inch),
        (8.3*inch, H-3.8*inch, 8.3*inch, H-5.0*inch),
        (10.5*inch, H-3.8*inch, 10.5*inch, H-5.0*inch),
    ]:
        c.line(x1, y1, x2, y2)
    c.setFillColor(NAVY)
    c.setFont('Helvetica-Bold', 13)
    c.drawString(0.8*inch, 1.8*inch, '資料流：使用者輸入 → 解析產品資料 → MCP Client 調用工具 → RAG 檢索官方文件 → 規則引擎判讀 → LLM 整理結論與依據 → UI 顯示/下載報告')


def add_page4(c):
    draw_header(c, '三、系統設計與模組分工', '設計重點：模組化、可追蹤、可驗證與可擴充')
    draw_bullets(c, 0.8*inch, H - 2.1*inch, 5.0*inch, [
        'app.py：Streamlit 互動介面，負責產品輸入、顯示判讀結果與法規來源。',
        'foodguard/：負責輸入解析、產品資料結構化、規則判讀與上下文處理。',
        'backend/：處理 PDF 讀取、chunk 切分、embedding 與索引建立。',
        'rag/：建立 FAISS 向量儲存與檢索，保留 chunk metadata、source、page 與分數。'
    ], font_size=12)
    draw_bullets(c, 6.4*inch, H - 2.1*inch, 5.0*inch, [
        'mcp_server.py：對外提供 search_food_regulation、check_allergens 等工具。',
        'mcp_client.py：負責與 MCP Server 通訊，統整 tool responses 與會話歷史。',
        'data/：保存 FAISS index、SQLite 記憶體、報告與規則資料。',
        'documents/：官方法規資料來源，作為唯一判斷依據來源。'
    ], font_size=12)


def add_page5(c):
    draw_header(c, '四、設計原則', '系統設計遵守可驗證性與安全性')
    principles = [
        '法規唯一來源：法規依據僅取自 documents/，不自行生成法規內容。',
        '可追溯證據：每個判定保留文件名稱、頁碼、chunk ID、引用內容與相關度。',
        '規則優先：數字比較與欄位判斷由 Python rule engine 執行。',
        '失敗時保守：若無足夠證據，統一顯示「目前知識庫找不到足夠依據」。',
        'MCP 協作：工具分工明確，讓檢索、核對與摘要分離。',
        '多層 fallback：產品資料、DRIs、RAG 與 LLM 可作為保底，但不取代法規依據。',
    ]
    for i, text in enumerate(principles):
        y = H - 2.0*inch - i*0.75*inch
        c.setFillColor(LIGHT)
        c.setStrokeColor(BLUE)
        c.setLineWidth(1)
        c.roundRect(0.7*inch, y, 11.5*inch, 0.52*inch, 10, fill=1, stroke=1)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 11)
        c.drawString(0.9*inch, y + 0.15*inch, text)


def add_page6(c):
    draw_header(c, '五、系統規格', '功能、環境與執行需求')
    specs = [
        ('開發環境', ['Windows / VS Code', 'Python 3.11', 'Streamlit', 'Official MCP Python SDK', 'OpenAI API / Ollama']),
        ('核心資料', ['documents/ 法規文件', 'FAISS 向量索引', 'SQLite session memory', 'nutrition_claim_rules.json', 'food_composition.db']),
        ('核心能力', ['營養宣稱查核', '過敏原辨識', '法規 RAG 檢索', 'Web Search / 問答', '離線報告輸出'])
    ]
    x_positions = [0.8*inch, 4.9*inch, 9.0*inch]
    for idx, (title, items) in enumerate(specs):
        x = x_positions[idx]
        c.setFillColor(LIGHT)
        c.setStrokeColor(BLUE)
        c.setLineWidth(1)
        c.roundRect(x, H - 3.3*inch, 3.5*inch, 2.3*inch, 12, fill=1, stroke=1)
        c.setFillColor(BLUE)
        c.setFont('Helvetica-Bold', 14)
        c.drawString(x + 0.2*inch, H - 1.8*inch, title)
        c.setFillColor(WHITE)
        c.setFont('Helvetica', 11)
        y = H - 2.35*inch
        for item in items:
            c.drawString(x + 0.25*inch, y, '• ' + item)
            y -= 0.25*inch
    c.setFillColor(NAVY)
    c.setFont('Helvetica', 12)
    c.drawString(0.9*inch, 1.6*inch, '系統支援本機運行與 Web UI 展示，並可將結果保存至 SQLite 與 HTML 報告檔。')
    c.drawString(0.9*inch, 1.3*inch, '資料處理流程以 deterministic rule engine 為主，並以 LLM 做整合式摘要與說明。')


def add_page7(c):
    draw_header(c, '六、系統工作流', '由輸入到判讀結論的完整執行流程')
    steps = [
        ('1. 輸入資料', '使用者輸入品名、成分、營養標示與宣稱。'),
        ('2. 解析產品', 'foodguard.parsing 正規化為 product_data 結構。'),
        ('3. 呼叫工具', 'MCP Client 透過 stdio 啟動 MCP Server。'),
        ('4. 檢索法規', 'RAG 搜尋相關文件與依據來源。'),
        ('5. 進行判讀', 'rule engine 比對欄位、門檻值與宣稱適用性。'),
        ('6. 輸出結果', 'UI 顯示結論、引證、建議與報告下載。')
    ]
    positions = [(0.75*inch, H - 3.1*inch), (4.1*inch, H - 3.1*inch), (7.45*inch, H - 3.1*inch), (0.75*inch, H - 5.1*inch), (4.1*inch, H - 5.1*inch), (7.45*inch, H - 5.1*inch)]
    for (x, y), (title, desc) in zip(positions, steps):
        c.setFillColor(LIGHT)
        c.setStrokeColor(BLUE)
        c.setLineWidth(1)
        c.roundRect(x, y, 2.8*inch, 1.1*inch, 12, fill=1, stroke=1)
        c.setFillColor(NAVY)
        c.setFont('Helvetica-Bold', 12)
        c.drawString(x + 0.15*inch, y + 0.7*inch, title)
        c.setFont('Helvetica', 9)
        c.drawString(x + 0.15*inch, y + 0.2*inch, desc)


def add_page8(c):
    draw_header(c, '七、結論與未來發展', '系統可作為食品標示智慧檢查原型，兼顧可用性與法規可信度')
    left_lines = [
        '本系統能將食品標示判讀從純文字問答，轉為具法規依據與可追溯的結構化判讀。',
        '透過 MCP 將檢索、規則與摘要模組拆分，有助於維護與延展。',
        'Faiss + RAG + 公式規則結合，使判讀更具可信度與一致性。',
        '未來可以擴大資料涵蓋範圍，整合更多官方食品法規與數據庫。'
    ]
    right_lines = [
        '可加入更多食品類別與營養項目判讀邏輯。',
        '可整合雲端部署與使用者帳號管理。',
        '可強化 Web Search + FAQ + 風險提醒功能。',
        '可發展為實際合規審核工具或校內研究展示平台。'
    ]
    draw_bullets(c, 0.8*inch, H - 2.5*inch, 5.0*inch, left_lines, font_size=12)
    draw_bullets(c, 6.5*inch, H - 2.5*inch, 5.0*inch, right_lines, font_size=12)
    c.setFillColor(GRAY)
    c.setFont('Helvetica', 11)
    c.drawString(0.9*inch, 0.55*inch, 'FoodGuard：結合 MCP、RAG、規則引擎與 Streamlit 的食品標示智慧判讀架構')


output = 'FoodGuard_系統架構與規格簡報.pdf'
canvas_doc = canvas.Canvas(output, pagesize=landscape(letter))
for page in [add_page1, add_page2, add_page3, add_page4, add_page5, add_page6, add_page7, add_page8]:
    page(canvas_doc)
    canvas_doc.showPage()
canvas_doc.save()
print(f'PDF created: {output}')
