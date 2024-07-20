
## Introduction

ในโปรเจกต์นี้ เราจะอธิบายขั้นตอนการทำ Tokenization, Encoding และ Embedding สำหรับข้อความในงานประมวลผลภาษาธรรมชาติ (NLP) โดยใช้ Hugging Face Transformers library กระบวนการนี้เป็นขั้นตอนสำคัญที่ช่วยแปลงข้อความจากรูปแบบที่มนุษย์เข้าใจได้เป็นรูปแบบที่คอมพิวเตอร์สามารถประมวลผลได้

# NLP Tokenization, Encoding, and Embedding with Hugging Face Transformers

ในงานประมวลผลภาษาธรรมชาติ (Natural Language Processing หรือ NLP) มีหลายขั้นตอนที่ใช้ในการแปลงข้อความจากรูปแบบที่มนุษย์เข้าใจได้ให้เป็นรูปแบบที่คอมพิวเตอร์สามารถประมวลผลได้ ขั้นตอนหลักๆ ที่คุณกล่าวถึงคือ Encoding, Tokenization และ Embedding ซึ่งมีบทบาทและความสำคัญแตกต่างกันไป ดังนี้:

## Tokenization:

ความหมาย: เป็นกระบวนการแบ่งข้อความออกเป็นหน่วยย่อยที่เรียกว่า "โทเค็น" (tokens) ซึ่งอาจจะเป็นคำ (words), ประโยค (sentences), หรือแม้แต่ตัวอักษร (characters) ขึ้นอยู่กับบริบทและเป้าหมายของงานนั้นๆ
ตัวอย่าง: ประโยค "สวัสดีครับ ผมชื่อสมชาย" อาจถูก tokenized เป็น ["สวัสดีครับ", "ผม", "ชื่อ", "สมชาย"]
## Encoding:

ความหมาย: เป็นกระบวนการแปลงโทเค็นที่ได้จากการ tokenization ให้เป็นค่าที่คอมพิวเตอร์สามารถเข้าใจและประมวลผลได้ เช่น ตัวเลข
ตัวอย่าง: โทเค็น ["สวัสดีครับ", "ผม", "ชื่อ", "สมชาย"] อาจถูก encoded เป็น [1, 2, 3, 4] โดยใช้พจนานุกรมที่แม็ปคำกับตัวเลข
## Embedding:

ความหมาย: เป็นการแปลงโทเค็นที่ถูก encode เป็นค่าตัวเลขให้เป็นเวกเตอร์ที่มีมิติหลายๆ มิติ ซึ่งจะช่วยในการจับลักษณะหรือความหมายของคำในบริบทต่างๆ
ตัวอย่าง: โทเค็น "สมชาย" อาจถูกแทนด้วยเวกเตอร์เช่น [0.1, 0.3, 0.7, ...] โดยใช้เทคนิคเช่น Word2Vec, GloVe หรือ BERT ที่สามารถจับความหมายและความสัมพันธ์ระหว่างคำได้ดีกว่าเพียงแค่ตัวเลขธรรมดา


# Detailed Process
## 1. Tokenization
Tokenization เป็นกระบวนการแปลงข้อความเป็นโทเค็น (tokens) ซึ่งเป็นหน่วยที่เล็กที่สุดที่โมเดลสามารถประมวลผลได้ โดยใช้ tokenizer จาก Hugging Face Transformers library
ซึ่งเป็นการทำ tokenizer และ encoder ในคำสั่งเดียวนั้นคือ tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```bash
from transformers import AutoTokenizer

# Load the tokenizer
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenize the input text
text = "Hello, my name is John."
inputs = tokenizer(text, return_tensors="pt")

# Display tokenized input
print(inputs)

```

```bash
{
    'input_ids': tensor([[  101,  7592,  1010,  2026,  2171,  2003,  2198,  1012,   102]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```
ในขั้นตอนนี้ ข้อความจะถูกแปลงเป็นโทเค็นและแทนด้วยตัวเลขที่เรียกว่า input_ids นอกจากนี้ยังมี attention_mask ซึ่งใช้ในการระบุว่าโทเค็นใดที่เป็นส่วนหนึ่งของข้อความและโทเค็นใดที่เป็น padding

## 2. Encoding
Encoding เป็นกระบวนการแปลงโทเค็นให้เป็นตัวเลข (input_ids) ที่คอมพิวเตอร์สามารถเข้าใจได้ โดยขั้นตอนนี้จะถูกดำเนินการพร้อมกับ Tokenization อย่างที่กล่าวไปเมื่อข้อที่แล้ว

## 3. Embedding
Embedding เป็นกระบวนการแปลงตัวเลข (input_ids) ให้เป็นเวกเตอร์ที่มีมิติหลายๆ มิติ ซึ่งเวกเตอร์เหล่านี้จะเก็บข้อมูลเกี่ยวกับความหมายและความสัมพันธ์ของคำในข้อความนั้นๆ โมเดลใน Hugging Face Transformers library จะทำการ Embedding โดยอัตโนมัติเมื่อทำการประมวลผลข้อมูล
```bash
from transformers import AutoModelForQuestionAnswering

# Load the model
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

# Perform forward pass to get embeddings and other outputs
outputs = model(input_ids=input_ids)

# Display model outputs
print(outputs)
```
```bash
QuestionAnsweringModelOutput(
    loss=None,
    start_logits=tensor([[-0.0862, -0.1346,  0.0669,  0.1735, -0.1573,  0.0515,  0.1061, -0.1576, -0.2093]]),
    end_logits=tensor([[-0.0732, -0.1431,  0.1191,  0.1781, -0.1491,  0.0582,  0.1076, -0.1413, -0.2223]])
)

```

### Conclusion
กระบวนการ Tokenization, Encoding และ Embedding เป็นขั้นตอนสำคัญในการเตรียมข้อมูลสำหรับการประมวลผลภาษาธรรมชาติ (NLP) โดยการใช้ Hugging Face Transformers library ทำให้กระบวนการเหล่านี้สะดวกและง่ายยิ่งขึ้น โดยไม่ต้องทำการ Embedding ด้วยตนเอง โมเดลจะจัดการเรื่องนี้ให้โดยอัตโนมัติเมื่อประมวลผลข้อมูล

# Sentence representation
 คือการแปลงประโยคหรือข้อความให้อยู่ในรูปแบบเวกเตอร์หรือตัวแทนเชิงตัวเลขที่คอมพิวเตอร์สามารถประมวลผลได้

## การทำ Sentence Representation ประกอบด้วยขั้นตอนหลักๆ ดังนี้:
### 1. Tokenization
การแบ่งข้อความออกเป็นโทเค็น (tokens) เช่น คำหรือวลี

### 2. Embedding
การแปลงโทเค็นเหล่านั้นให้เป็นเวกเตอร์เชิงตัวเลข (word embeddings) เช่น Word2Vec, GloVe, หรือการใช้ Embedding Layer ในโมเดล Transformer

### 3. Aggregation ( Pooling)
การรวมเวกเตอร์ของโทเค็นต่างๆ เข้าด้วยกันเพื่อสร้างเวกเตอร์ที่เป็นตัวแทนของประโยคทั้งหมด เช่น การใช้การเฉลี่ย (average) การใช้ LSTM หรือการใช้ Attention Mechanism

## หลักการทำงาน?
เมื่อเรากำหนด language model ขึ้นมา (ในรูปเป็น BERT) เมื่อป้อน input เข้าไปแล้ว output จาก BERT จะออกมาเป็น embedding ของคำแต่ละคำ จากนั้นเราจะทำการรวม embedding หรือ pooling เพื่อสร้าง sentence representation


![image](https://github.com/user-attachments/assets/bac939f1-53e8-41e4-8f9c-c81ec37abaab)


## ข้อดี?
ทำงานได้กับทุกโมเดลของ Hugging Face 100%
โมเดล sentence representation ที่ดีที่สุดในขณะนี้ก็ถูกสร้างบน sentence-transformer
ง่าย ไม่ว่าจะเทรนหรือใช้งาน

## จะให้โมเดลเรียนรู้ประโยคยังไง?
## คำตอบ: ใช้ท่าจากเปเปอร์ => SimCSE:Simple Contrastive Learning of Sentence Embeddings


![image](https://github.com/user-attachments/assets/286258c9-28d5-47fe-9c9f-29be3fec2e43)



## ทำไมต้อง SimCSE?
เทรนง่าย ไม่ต้อง setup อะไรยุ่งยาก
ใช้งานได้กับทุกโมเดล ทุกภาษา
ไม่จำเป็นต้องมี label (Unsupervised Learning)
ประสิทธิภาพเทียบเท่ากับ Supervised Learning

### การทำ    train_dataset 

ท่า SimCSE เราไม่ต้องพึ่ง คำสั่ง data argumentation จากภาย นอกเนื่องจากใน tranformers จะมี drop-out อยู่ ซึ่งตอนtrain Bert ก็จะทำให้ drop-out ทำงาน
    # ซึ่งจาก paper SimCSE บอกว่าใช้แค่ drop-out พวกนนี้ และโยน sentence เดียวกันเขาไปใน ข้อมูล (input example) สำหรับการฝึกโมเดล ก็สามารถสร้าง vector ได้

```bash
with open(file=os.path.join("wiki_20210620_clean.txt"),mode="r",encoding="utf-8") as f:
    raw_dataset = f.readlines()[:100000]
train_dataset = [sentence_transformers.InputExample(texts=[sent,sent]) for sent in raw_dataset]
```
## Contrastive (InfoNCE) Loss 
 Contrastive Loss  ถูกใช้เพื่อให้โมเดลเรียนรู้การแทนที่ (Embedding Learning) หรือ กระบวนการที่ใช้ในการแปลงข้อมูล ให้เป็นเวกเตอร์ที่มีมิติที่ต่ำกว่า 
 ซึ่งสามารถจับลักษณะเชิงความหมายหรือคุณสมบัติสำคัญของข้อมูลที่สนใจ (anchor)
 ที่มีลักษณะคล้ายกันสำหรับตัวอย่างที่คล้ายกัน (positive) และลักษณะต่างกันสำหรับตัวอย่างที่แตกต่างกัน (negative)
 
 InfoNCE Loss ใช้ในงานที่ต้องการเรียนรู้การ embadding ที่มีความหมาย โดยการเปรียบเทียบการแทนที่ที่ คล้ายกันและไม่คล้าย กันในกลุ่มตัวอย่าง 
        # ตัวอย่างเช่น ในงานที่เกี่ยวข้องกับการจับคู่ข้อมูลหรือการเรียนรู้เชิงความคล้ายคลึง 



![image](https://github.com/user-attachments/assets/ab52b2f5-4524-456c-a5a8-27af0362d0d5)



### ตัวแปรแต่ละตัวหมายถึง?
z_i = anchor (ประโยคที่สนใจ) สามารถสุ่มมาจาก training data ได้เลย
z_j = positive (ประโยคที่คู่กับ anchor) ซึ่งในที่นี้เราสร้างจาก data augmentation
z_k = negative (ประโยคที่ไม่ได้คู่กับ anchor) ในที่นี้หมายถึงประโยคที่เหลือ (ที่ไม่ใช่ anchor กับ positive) 

### จุดประสงค์ของ Contrastive?
* เราต้องการให้ z_i กับ z_j มี similarity มากที่สุด
* ขณะเดียวกัน เราต้องการให้ z_i กับประโยคที่เหลือ มี similarity น้อยที่สุด

## ภาพรวม


![image](https://github.com/user-attachments/assets/b3045613-9298-449c-a498-53086593340b)



## Evaluation: STS data
เป็นดาต้าเซ็ตคู่ ที่เอาไว้บ่งบอกว่าประโยคแรก และ ประโยคที่สองมีความสัมพันธ์กันอย่างไรเช่น "ผู้ชายกำลังเล่นพิณ" กับ "ผู้ชายกำลังเล่นแป้นพิมพ์" จะมีความสัมพันธ์กันอยู่ที่ 1.5 คะแนน (เต็ม 5) ซึ่งคะแนนเหล่านี้มาจากกฏของภาษาศาสตร์
