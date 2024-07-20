######################################################################################  Intro  ######################################################################################

# Sentence representation

# คือการแปลงประโยคหรือข้อความให้อยู่ในรูปแบบเวกเตอร์หรือตัวแทนเชิงตัวเลขที่คอมพิวเตอร์สามารถประมวลผลได้

# การทำ Sentence Representation ประกอบด้วยขั้นตอนหลักๆ ดังนี้:

# 1. Tokenization
# การแบ่งข้อความออกเป็นโทเค็น (tokens) เช่น คำหรือวลี

# 2. Embedding
# การแปลงโทเค็นเหล่านั้นให้เป็นเวกเตอร์เชิงตัวเลข (word embeddings) เช่น Word2Vec, GloVe, หรือการใช้ Embedding Layer ในโมเดล Transformer

# 3. Aggregation ( Pooling)
# การรวมเวกเตอร์ของโทเค็นต่างๆ เข้าด้วยกันเพื่อสร้างเวกเตอร์ที่เป็นตัวแทนของประโยคทั้งหมด เช่น การใช้การเฉลี่ย (average) การใช้ LSTM หรือการใช้ Attention Mechanism

#########################################################################  Create model  #####################################################################################

import sentence_transformers
import sentence_transformers.evaluation
import sentence_transformers.losses
import sentence_transformers.models
import os
import torch
import torch.utils
import torch.utils.data
import  pandas as pd
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Sentence-transformer เป็น library สำหรับสร้าง/เทรน sentence representation จาก transformer (ซึ่งรองรับทุกโมเดลบน Hugging Face)
#  เพื่อสร้างเวกเตอร์ที่เป็นตัวแทนของประโยค (sentence embeddings) สำหรับการคำนวณความคล้ายคลึงระหว่างประโยค:
model_checkpoint_name =  'airesearch/wangchanberta-base-att-spm-uncased'

word_embadding_model = sentence_transformers.models.Transformer(model_name_or_path=model_checkpoint_name,max_seq_length=32)
    # คำสั่งนี้เป็นการสร้างโมเดลสำหรับการสร้างเวกเตอร์ที่เป็นตัวแทนของประโยค
    # โดยใช้โมเดล Transformer ที่ชื่อว่า airesearch/wangchanberta-base-att-spm-uncased เป็นตัวสร้างเวกเตอร์คำ (word embeddings)
    # max_seq_length=32 กำหนดความยาวสูงสุดของลำดับโทเค็นที่ประมวลผล โดยถ้าประโยคยาวกว่านี้จะถูกตัดให้เหลือเพียง 32 โทเค็น

pooling_model = sentence_transformers.models.Pooling(word_embedding_dimension=word_embadding_model.get_word_embedding_dimension(),pooling_mode="cls")
    # คำสั่งนี้เป็นการสร้างโมเดลสำหรับการทำ Pooling
    # word_embedding_dimension เป็นขนาดของเวกเตอร์ที่ได้จาก word_embedding_model
    # pooling_mode="cls" กำหนดให้ใช้เวกเตอร์ที่ได้จากโทเค็นพิเศษ [CLS] เป็นตัวแทนของประโยค

model = sentence_transformers.SentenceTransformer(modules=[word_embadding_model,pooling_model])
    # คำสั่งนี้เป็นการสร้างโมเดล SentenceTransformer โดยรวมโมดูล word_embedding_model และ pooling_model เข้าด้วยกัน
    # เพื่อสร้างเวกเตอร์ที่เป็นตัวแทนของประโยค (sentence embeddings) สำหรับการคำนวณความคล้ายคลึงระหว่างประโยค:

# ใช้ท่าจากเปเปอร์ => SimCSE:Simple Contrastive Learning of Sentence Embeddings

#########################################################################  read dataset  #####################################################################################

with open(file=os.path.join("wiki_20210620_clean.txt"),mode="r",encoding="utf-8") as f:
    raw_dataset = f.readlines()[:100000]


train_dataset = [sentence_transformers.InputExample(texts=[sent,sent]) for sent in raw_dataset]
    # ใช้ในการสร้างชุดข้อมูลสำหรับการฝึกโมเดล
    # sentence_transformers.InputExample : เป็นคลาสที่ใช้ในการสร้างตัวอย่างข้อมูล (input example) สำหรับการฝึกโมเดล
    # ในกรณีนี้, [sent, sent] หมายความว่าประโยคเดียว (sent) ถูกนำมาใช้เป็นข้อมูลฝึกในสองกรณี (pair) คือ ประโยคแรกและประโยคที่สองในตัวอย่างการฝึก
        # แต่ใน ท่า SimCSE เราไม่ต้องพึ่ง คำสั่ง data argumentation จากภาย นอกเนื่องจากใน tranformers จะมี drop-out อยู่ ซึ่งตอนtrain Bert ก็จะทำให้ drop-out ทำงาน
        # ซึ่งจาก paper SimCSE บอกว่าใช้แค่ drop-out พวกนนี้ และโยน sentence เดียวกันเขาไปใน ข้อมูล (input example) สำหรับการฝึกโมเดล ก็สามารถสร้าง vector ได้
    # ทำการใส่ training data เข้าไปใน

# print("\n\n")
# print(train_dataset[:5])
# print("\n\n")

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,num_workers=0,batch_size=32,shuffle=True)

# print("\n\n")
# print(train_dataloader)
# print("\n\n")

loss_function =  sentence_transformers.losses.MultipleNegativesRankingLoss(model=model) # Contrastive Loss
        # Contrastive(InfoNCE) Loss :เป็นฟังก์ชันการสูญเสียที่ใช้ในงานการเรียนรู้การแทนที่ (representation learning) และการเรียนรู้ที่มีความคล้ายคลึงกัน (contrastive learning) 

        # InfoNCE Loss ใช้ในงานที่ต้องการเรียนรู้การ embadding ที่มีความหมาย โดยการเปรียบเทียบการแทนที่ที่ คล้ายกันและไม่คล้าย กันในกลุ่มตัวอย่าง 
        # ตัวอย่างเช่น ในงานที่เกี่ยวข้องกับการจับคู่ข้อมูลหรือการเรียนรู้เชิงความคล้ายคลึง 

        #  ถูกใช้เพื่อให้โมเดลเรียนรู้การแทนที่ (embadding Learning) หรือ กระบวนการที่ใช้ในการแปลงข้อมูล ให้เป็นเวกเตอร์ที่มีมิติที่ต่ำกว่า 
            # ซึ่งสามารถจับลักษณะเชิงความหมายหรือคุณสมบัติสำคัญของข้อมูลที่สนใจ (anchor)
            # ที่มีลักษณะคล้ายกันสำหรับตัวอย่างที่คล้ายกัน (positive) และลักษณะต่างกันสำหรับตัวอย่างที่แตกต่างกัน (negative)

        # Contrastive Loss เป็นวิธีที่มีประสิทธิภาพในการฝึกโมเดลให้สามารถแยกแยะและจับคู่ตัวอย่างได้ดีขึ้น 

##################################################################################### 

#############################################################################  evaludate model with  STS data  ##################################################################

# เป็นดาต้าเซ็ตคู่ ที่เอาไว้บ่งบอกว่าประโยคแรก และ ประโยคที่สองมีความสัมพันธ์กันอย่างไรเช่น "ผู้ชายกำลังเล่นพิณ" กับ "ผู้ชายกำลังเล่นแป้นพิมพ์" จะมีความสัมพันธ์กันอยู่ที่ 1.5 คะแนน (เต็ม 5) 
# ซึ่งคะแนนเหล่านี้มาจากกฏของภาษาศาสตร์

df = pd.read_csv(os.path.join(r"D:\machine_learning_AI_Builders\บท4\NLP\Text_Sentence_Embedding\sts-test_th.csv"),header=None)
test_data = df.dropna(inplace=False).values.tolist()
    # .dropna() ใช้สำหรับลบแถว (rows) หรือคอลัมน์ (columns) ที่มีค่าที่เป็น NaN (Not a Number) หรือค่าที่หายไป (missing values) ออกไปจาก DataFrame หรือ Series ที่ใช้เมธอดนี้
        # ค่าเริ่มต้น, .dropna() จะลบแถวที่มีค่า NaN หรือค่าที่หายไปใน DataFrame

    # inplace=True  จะทำให้การดำเนินการเปลี่ยนแปลงเกิดขึ้นใน DataFrame หรือ Series ต้นฉบับโดยตรง แทนที่จะสร้าง DataFrame หรือ Series ใหม่และคืนค่าผลลัพธ์ใหม่

    #.values ใช้สำหรับดึงข้อมูลออกจาก DataFrame หรือ Series และแปลงข้อมูลนั้นเป็นอาเรย์ numpy ซึ่งเป็นการแทนที่ข้อมูลในรูปแบบของอาเรย์ numpy
    #  ที่สามารถใช้ในการคำนวณทางคณิตศาสตร์หรือประมวลผลข้อมูลที่ต้องการการทำงานระดับต่ำ


test_samples = []
for row in test_data:
    score = float(row[2]) / 5.0  # Normalize score to range 0 ... 1
    test_samples.append(sentence_transformers.InputExample(texts=[row[0], row[1]], label=score)) #

test_evaluator = sentence_transformers.evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=32, name='sts-test')

print(f'Befor Train model score {test_evaluator(model)["sts-test_spearman_max"]}')
print("\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\tV")
#print(f'After Train model score {test_evaluator(model)}')

# model.fit(train_objectives=[(train_dataloader,loss_function)],
#           epochs=1,
#           show_progress_bar=True,
#           save_best_model=True,
#           output_path="Sentence_Embedding_SimCSE_wangchanberta_finetune_wiki",
#           optimizer_params={"lr":3e-5},
#           )

print(f'Afer Train model score {test_evaluator(model)["sts-test_spearman_max"]}')
print("\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\tV")

a_vec = model.encode(['วันนี้อากาศดีมาก','ฉันกินข้าวแล้ว','Bad weather','Im not hungry'],normalize_embeddings=True)
b = model.encode(['วันนี้อากาศดี'],normalize_embeddings=True)
print(np.inner(a_vec,b))