import numpy as np
import gradio as gr
import torch
import utils.generator




if torch.cuda.is_available():
    device = torch.device("cuda")
    Tensor = torch.cuda.FloatTensor
else:
    device = torch.device("cpu") 
    Tensor = torch.FloatTensor





################### GAN 모델 호출 ###############################

from torch.autograd import Variable
from torchvision.transforms.functional import to_pil_image

GANmodel = utils.generator.Generator(latent_dim=100,image_shape=(3,224,224))

GANmodel.load_state_dict(torch.load("generator.pth", map_location=device))

GANmodel.eval()

def making_img(output):

    if output != "옷":
        raise gr.Error("죄송합니다. 시간 관계상 모델 학습을 못시켰습니다. ㅠㅠ")

    if output == "옷":

        z = Variable(Tensor(np.random.normal(0, 1, (1, 100))))

        gen_images = GANmodel(z)

        img = gen_images[0]

        img = to_pil_image(0.5*img+0.5)

        imgArray = np.array(img) # 넘파이로 변경
        
        return imgArray

#################################################################







################## ResNet 호출 ##########################################
from utils.train_to_test_utils import set_augmentations
from utils.gui_utils import get_label_dict, get_model, preprocess_image

model = get_model(device)
label_dict = get_label_dict()

def predict(input_image):
    transforms = set_augmentations('test')
    image = preprocess_image(input_image, transforms)

    model.eval()
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(image)[0], dim=0)
        confidences = {label_dict[i]: float(prediction[i]) for i in range(12)}

        return confidences


def predict2(input_image):
    transforms = set_augmentations('test')
    image = preprocess_image(input_image, transforms)

    model.eval()
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(image)[0], dim=0)
        confidences = {label_dict[i]: float(prediction[i]) for i in range(12)}

        return max(confidences, key = confidences.get)

############################################################################







# 사진찍으면 멈추기!
def stop(inp):
    return inp


# 총 리스트
label_list =['battery','biological','brown-glass','cardboard','clothes','green-glass','metal','paper','plastic','shoes','trash','white-glass']

# 라디오 뭐 눌렸냐에따라 보여줄거 설정 
def showInfor(output,drop3):

    if output not in label_list:
        return "# 이미지를 넣어서 정보를 확인해주세요! ^_^"

    # 찾아보기가 눌리면 기본값으로 나옴.
    if drop3 == "찾아보기":
        return """
        # (김해) 지정 민간 재활용 센터
        |센터명|전화번호|소재지|취급품목|
        |:---:|:---:|:---:|:---:|
        |바다재활용백화점|055-327-8964|경상남도 김해시 분성로450 (삼정동)|가전, 가구, 사무용기자재, 생활용품 등|
        |구구재활용센터|055-326-2624|경상남도 김해시 분성로 456(삼정동)|가전, 가구, 사무용기자재, 생활용품 등|
        |김해대형종합재활용|055-327-7272|경상남도 김해시 진례면 서부로 480|가전, 가구, 사무용기자재, 생활용품 등|
        |(사)열린정보장애인협회<br>정부물품 재활용센터|055-325-0444|경상남도 김해시 주촌면 서부로1701번안길 58-47|가전, 가구, 사무용기자재, 생활용품 등|

        - 취급품목 : 가구류, 가전제품류, 사무기기류 등
        - 수거방법 : 방문 수거(무상 또는 유상 매입)


        <span style="color:red"> ※ 판매가능한 제품만 수거(재활용센터로 문의)</span>
        <img src="https://www.gimhae.go.kr/_res/depart/img/sub/p01721_img4.jpg"/>
        
        # 기타 민간 재활용 센터
        |구분|재활용센터명|소재지|전화번호|취급품목|
        |:---:|:---:|:---:|:---:|:---:|
        |1|경남재활용센터|경상남도 김해시 진영읍 여래로 6-4|055-346-0408|가전제품|
        |2|초대형재활용|경상남도 김해시 분성로 60 (외동)|055-337-6144|가전제품, 가구|
        |3|내외동재활용|경상남도 김해시 분성로 108 (외동)|055-321-0161|가전제품, 가구|
        |4|동서재활용|경상남도 김해시 금관대로 1266-1 (내동)|055-325-7282|가전제품, 가구|
        |5|신도시재활용판매장|경상남도 김해시 생림대로 15 (삼계동)|055-332-6770|가전제품, 가구|
        |6|부림재활용수집소|경상남도 김해시 구지로124번길 21 (대성동)|055-333-5734|가전제품|
        |7|장유벼룩시장|경상남도 김해시 내덕로4번길 3 (무계동)|055-312-9701|가전제품, 가구|
        |8|김해재활용협동조합|경상남도 김해시 칠산로 251 (화목동)|055-322-7474|가전제품, 가구|
        |9|태양재활용|경상남도 김해시 칠산로251번길 6-12(화목동)|055-326-7474|가전제품|
        |10|대형재활용|경상남도 김해시 분성로 528 (어방동)|055-324-8002|가전제품, 가구|
        |11|김해상설재활용|경상남도 김해시 분성로 421 (삼정동)|055-332-9882|가전제품, 가구|
        |12|김해재활용마트|경상남도 김해시 분성로 462 (삼정동)|055-321-9604|가전제품, 가구|
        |13|어방재활용백화점|경상남도 김해시 분성로 497 (어방동)|055-329-4585|가전제품, 가구|
        |14|김해재활용매장|경상남도 김해시 인제로51번길 58-2 (삼정동)|055-339-7757|가전제품, 가구|
        |15|김해상설재활용센터|경상남도 김해시 분성로 705 (지내동)|055-333-3906|가전제품, 가구|
        |16|고개재활용프라자|경상남도 김해시 분성로 400 (동상동)|055-325-0192|가전제품, 가구|
        
        """

    if drop3 == "분리수거 방법":

        if output in ['paper','cardboard']:

            return """
            # 종이

            우리나라의 종이 사용량은 해마다 증가하고 있으며 이에 따른 폐지의 발생량도 늘어나고 있습니다. 

            2000년의 경우를 보면 우리 나라의 폐지 자급율은 69.6%에 불과하고 수입종이 금액은 약 4,090억원에 달함으로 철저한 분리수거로 종이 수입 의존도를 줄여야 겠습니다.

            # 분리배출 요령

            오물이나 물에 젖지 않도록하고 비닐, 플라스틱, 알루미늄, 철사 등 이물질이 섞이지 않도록 해야 합니다.

            <img src="https://www.gimhae.go.kr/_res/depart/img/sub/p01745_img2.gif"/>

            |분류|방법|
            |:---:|:---:|
            |신문지|> 물기에 젖지 않도록 하고 반듯하게 펴서 차곡 차곡 쌓은후 30㎝ 정도의 높이로 끈으로 묶어 배출 <br> > 비닐코팅된 광고지, 비닐류, 기타 오물이 섞이지 않도록 함 |
            |책자, 노트, 종이쇼핑백, 달력, 포장지|> 비닐로 코팅된 표지, 공책의 스프링 등은 재활용이 되지 않음|
            |우유팩, 음료수팩, 종이컵|> 내용물을 비운 뒤 물로 한번 헹군 후 압축하여 봉투에 넣거나 다른팩에 5∼6개씩 넣거나 펴서 말린 후 배출|
            |상자류(과자, 포장상자, 기타 골판지상자 등)|>상자에 붙어있는 테이프, 철핀 등을 제거한 후 압착하여|
            """

        if output in ['plastic']:

            return """
            # 플라스틱

            가공이 쉽고 녹슬지 않으며 내구성이 양호한 플라스틱은 석유공업의 발달과 생활의 편리성 추구로 사용량이 많은 반면 자연분해되지 않아 매립해도 오랫동안 썩지 않고 그대로 남아있게 됩니다. 
            소각시에는 완전연소가 어렵고 유독가스를 발생시키며 소각 후에도 중금속의 잔재가 남기 때문에 단순 매립할 경우 2차적인 환경오염을 일으키게 됩니다. 
            그러므로 폐플라스틱의 처리는 재활용하는 것이 가장 효과적입니다.
            
            ![image.jpg1](https://www.gimhae.go.kr/_res/depart/img/sub/p01746_img1.png) |![image.jpg2](https://www.gimhae.go.kr/_res/depart/img/sub/p01746_img2.png)|![image.jpg3](https://www.gimhae.go.kr/_res/depart/img/sub/p01746_img3.png)|![image.jpg4](https://www.gimhae.go.kr/_res/depart/img/sub/p01746_img4.png)|![image.jpg5](https://www.gimhae.go.kr/_res/depart/img/sub/p01746_img5.png)|![image.jpg6](https://www.gimhae.go.kr/_res/depart/img/sub/p01746_img6.png)|![image.jpg7](https://www.gimhae.go.kr/_res/depart/img/sub/p01746_img7.png)
            --- | --- | --- | --- | --- | --- | --- |

            # 분리배출 요령

            |분류|방법|
            |:---:|:---:|
            |PET, PVC, PP, PS, PE, PSP재질의 용기·포장재|> 내용물을 깨끗이 비우고 다른 재질로 된 뚜껑(또는 은박지, 랩 등)이나 부착상표 등을 제거한 후 가능한 압착하여 배출 |
            |스티로폼 완충재 <br> - 전자제품 완충재로 사용되는 발포합성 수지·포장재 <br> - 농·축산물 포장용 발포스티렌상자|> TV, 냉장고, 세탁기, 에어컨디셔너, 오디오, 개인용 컴퓨터, 이동전화 단말기 제품의 발포합성수지 완충재는 제품구입처로 반납 <br> > 내용물을 완전히 비우고 부착상표 등을 제거하고 이물질이 묻은 경우 깨끗이 씻어서 배출<br> > 음식물등 이물질이 많이 묻어 있거나 타물질로 코팅된 발포스티렌은 제외|

            ![image.jpg1](https://www.changwon.go.kr/depart/img/sub10/05/img_plastic01_01.jpg) |![image.jpg2](https://www.changwon.go.kr/depart/img/sub10/05/img_plastic01_02.jpg)|![image.jpg3](https://www.changwon.go.kr/depart/img/sub10/05/img_plastic01_03.jpg)
            | --- | --- | --- |
            | 용기의 표면 또는 바닥 부분에 표기된 분리배출표시 <br> (PEF, HDPE, PP,PS)를 확인하여 배출 | 뚜껑을 제거한 후 내용물을 비우고 가능한 압축하여 배출 | 부착 상표를 제거하여 배출 |
            """

        if output in ['brown-glass', 'green-glass','white-glass']:
            return """
            # 유리병

            생활의 편리함 추구로 인해 캔, 페트병 등과 같은 1회용품 사용이 늘어 왔으나 최근 들어 환경보호 차원에서 재활용을 위한 유리병 사용이 다시금 각광을 받고 있습니다. 
            유리병의 재활용은 크게 재사용과 원료 재활용으로 구분 할 수 있습니다. 
            먼저 재사용은 빈병을 회수하여 세척·소독 처리한 후 사용하는 것으로 빈용기보증금제도의 정착으로 90% 이상 활용되고 있습니다. 
            다음 원료 재활용은 깨뜨려서 유리제품의 원료로 사용하는 것인데 재활용률은 약 0% 수준입니다.
            
            ![image](https://www.gimhae.go.kr/_res/depart/img/sub/p01747_img1.gif)


            # 분리배출 요령

            - 플라스틱이나 알루미늄 뚜껑 제거
            - 내용물을 깨끗이 비운 후 물로 헹구어 되도록 무색, 청·녹·갈색으로 분리
            - 담배꽁초등 이물질을 넣지 말것
            
            
            <span style="color:red"> ※ 맥주병, 소주병, 청량음료병은 슈퍼에 되돌려주고 빈용기보증금을 환불 받을 수 있음</span>
            
            ![image](https://www.gimhae.go.kr/_res/depart/img/sub/p01747_img2.gif)
            
            """

        if output in ['metal']:
            return """
            # 캔

            한해 동안 사용되는 캔의 양은 약 6억개로 그 중 1.2억개가 알루미늄캔이며 나머지가 철캔입니다. 
            알루미늄캔을 재활용하는데 필요한 에너지는 원석으로부터 알루미늄을 얻는데 필요한 에너지의 1/26로 에너지 절약효과가 큽니다. 
            또한 알루미늄캔 하나가 땅속에 묻힌 후 분해되는데 걸리는 시간이 500년이나 되기 때문에 환경보호효과도 큽니다.

            |분류|방법|
            |:---:|:---:|
            |철캔, 알루미늄캔(음·식용류)| - 캔속에 들어있는 내용물을 깨끗이 비우고 물로 헹군 후 압축하여 배출<br> - 겉 또는 속의 플라스틱 뚜껑 등 제거 <br> - 담배꽁초 등 이물질을 넣지말것|
            |기타캔(부탄가스, 살충제용기)| - 구멍을 뚫어 내용물을 비운 후 배출|

            ![image](https://www.gimhae.go.kr/_res/depart/img/sub/p01748_img1.gif)

            # 고철

            철강업의 3대 기초원료(철광석, 원료탄, 고철) 중 하나인 고철은 전기로 제강, 신철 및 주물업계의 주요한 원료입니다. 
            고철은 현재까지 주로 고로에서 선철제조시 사용하였으나 제강기술의 발전으로 전기로에서 철강,합금철 제조시 사용할 수 있게 됨에 따라 고철 사용이 증가될 것으로 전망됩니다. 
            '99년의 경우 철강재 소비량 39,513천톤 중 고철 사용량은 15,891천톤으로 재활용율은 40.2%이며 고철사용량 중 수입량이 7,771톤으로 수입의존도가 48.9%에 달합니다.

            - 고철 : 공구류, 철판 등
            - 비철금속 : 양은류, 스텐류, 전선, 알루미늄, 샷시류

            ## 분리배출 요령

            - 이물질이 섞이지 않도록 한 후 봉투에 넣거나 끈으로 묶어서 배출
            - 플라스틱등 기타 재질이 많이 섞인 폐품은 금속성분이 있더라도 고철로 배출하면 안됨

            ![image](https://www.gimhae.go.kr/_res/depart/img/sub/p01749_img1.gif)

            """

        if output in ['clothes','shoes']:
            return """
            # 의류

            헌 의류의 재활용은 크게 두 가지로 나눌 수 있습니다. 
            면섬유의 경우에는 흡습성이 좋아 공업용 걸레로 활용되어 기름을 닦거나 기타 오물을 닦아 내는데 유용하게 쓰이고 있습니다. 
            그 외 올이 잘 풀리는 소재로 만든 것은 농업용 보온덮개, 방음·방수 소재 등으로 재활용되고 있습니다.

            # 분리배출 요령

            - 입을 만한 옷들은 깨끗이 빨아 이웃, 친척과 알뜰매장 등에서 서로 교환하여 입거나 불우이웃과 나누어 입읍시다.
            - 카펫, 가죽백, 구두, 기저귀 커버 등과 같이 복합소재 제품이 섞이지 않도록 한다.
            - 쓸만한 단추나 지퍼 등은 따로 떼어내어 보관
            - 물기에 젖지 않도록 포대등에 담거나 30㎝ 높이로 묶은 후 배출

            ![image](https://www.gimhae.go.kr/_res/depart/img/sub/p01751_img1.gif)

            """

        if output in ['biological']:
            return """
            # 음식물 쓰레기 버리는 법

            ## 1.비닐 봉지가 아닌, 뚜껑이 있는 용기 사용하기
            
            비닐봉지에 음식물 쓰레기를 모아 버리는 경우를 흔하게 볼 수 있습니다. 그러다 보면 음식물 쓰레기에 비닐이 함께 버려지는 경우가 발생하기 마련입니다.

            음식물 쓰레기에 비닐봉지가 섞여 들어가게 되면 재활용하기가 어려워집니다. 비닐봉지를 걸러내기가 쉽지 않아 사료나 퇴비로 재활용했을 때 그 안에 그대로 들어가는 경우도 발생합니다.

            ![image](https://i0.wp.com/greennews360.com/wp-content/uploads/2021/06/pexels-karolina-grabowska-4033162.jpg?w=1280&ssl=1)

            비닐봉지를 음식물 쓰레기와 섞이지 않게 따로 버린다 하더라도 악취를 풍기고 미세플라스틱 문제 등의 환경 오염을 일으키게 됩니다. 

            따라서 음식물 쓰레기를 모을 때는 뚜껑이 있는 용기를 사용하는 것이 좋습니다.

            ## 2. 이물질이 들어가지 않도록 하기

            음식물 쓰레기에 이물질이 유입될 경우 수거와 처리 과정에서 장비 고장을 일으켜 재활용하는 데 걸림돌이 될 수 있습니다. 또한 이물질을 선별해 내는 데 추가 인력과 비용을 투입해야합니다.

            ![image](https://i0.wp.com/greennews360.com/wp-content/uploads/2021/06/pexels-lukas-952478.jpg?w=1280&ssl=1)

            가장 문제가 되고 있는 비닐봉지 외에도 버리는 과정에 쉽게 섞여 들어갈 수 있는 과도, 포크, 나무젓가락, 이쑤시개 등은 각별히 주의하여 절때 음식물 쓰레기로 함께 버려지지 않도록 해야 합니다.
            
            ## 3. 물기를 제거하고 부피를 줄이기

            음식물 쓰레기를 버리기 전에 최대한 수분을 제거하고 부피를 줄여 주는 것이 좋습니다.
            
            물기를 꽉 짜거나 건조시켜 배출하면 음식물 쓰레기를 처리할 때 소모되는 에너지를 줄 일 수 있기 때문입니다.

            길이가 길거나 크기가 큰 음식물 쓰레기는 잘게 잘라 부피를 줄여 배출합니다.

            ## 4. 음식물 쓰레기가 아닌 일반 쓰레기 구별하기.
            ![image](https://i0.wp.com/greennews360.com/wp-content/uploads/2021/06/pexels-keegan-evans-105588-1.jpg?w=1280&ssl=1)
            이 프로그램을 통해 자동으로 음식물 쓰레기와 일반 쓰레기가 구별되지만, 각 지자체마다 재활용 방식이 조금씩 달라 음식물 쓰레기 기준에 조금씩 차이가 있을 수 있습니다.

            아래 내용은 일반적인 기준이니 참고만 하시고, 아파트의 아내문 또는 지자체 홈페이지를 참고하거나 주민센터에 문의해서 정확한 음식물 쓰레기 분류 기준을 확인해 보시길 바랍니다!

            - 복숭아, 살구, 자두, 감 등 과일의 딱딱한 씨는 분쇄하기 어렵고 기계 고장을 일으킬 수도 있어 일반 쓰레기로 버려야 합니다.
            - 땅콩, 밤, 호두 등 딱딱한 견과류의 껍질도 일반 쓰레기입니다.
            - 닭 뼈와 족발 뼈다귀는 일반 쓰레기로 종량제 봉투에 버려야 합니다.
            ![image](https://i0.wp.com/greennews360.com/wp-content/uploads/2021/06/pexels-engin-akyurt-1435904.jpg?w=1280&ssl=1)
            - 고추장, 된장 등의 장류는 염도가 높아 동물 사료나 퇴비로 만들기 적합하지 않습니다. 따라서 통째로 종량제 봉투에 담아 버리거나 물에 희석해서 버려야 합니다.

            - 파인애플과 수박의 껍질은 일반 쓰레기입니다.
            - 커피 찌꺼기, 한약재 찌꺼기, 차 티백도 종량제 봉투에 버립니다.
            - 고추의 꼭지, 대파, 쪽파, 마늘의 껍질과 뿌리, 미나리 뿌리, 옥수수 껍질, 메추리알과 계란의 껍질도 일반 쓰레기입니다.
            - 비계와 내장은 포화 지방이 많아 사료로 만들기 부적합하기 때문에 일반 쓰레기로 버립니다.
            - 게, 조개, 홍합, 굴, 소라, 전복, 꼬막 등 어패류의 껍데기와 생선 가시도 일반 쓰레기입니다.

            ## 5. 당장 버릴 수 없을 때 잘 보관하는 법
            음식물 쓰레기는 바로 버리는 것이 가장 바람직하지만 그렇지 못할 경우 냉동실에 얼려 보관하는 분들이 종종 있습니다. 음식물 쓰레기를 냉동실에 보관하게 되면 저온성 세균과 바이러스가 증식해 식중독을 일으킬 수 있기 때문에 이 방법은 피하는 것이 좋습니다.

            ![image](https://i0.wp.com/greennews360.com/wp-content/uploads/2021/06/pexels-kaboompics-com-5765.jpg?w=1280&ssl=1)

            냉장고에 넣는 대신, 음식물 쓰레기에 베이킹 소다나 녹차가루를 뿌리면 악취를 제거하는 데 도움이 됩니다. 물과 소주를 3대 1로 섞어 뿌리면 날벌레가 생기는 것도 막을 수 있습니다.

            ## 6. 식용유, 상한 음식, 폐의의약품 버리는법
            ### 식용유
            ![image](https://i0.wp.com/greennews360.com/wp-content/uploads/2021/06/pexels-karolina-grabowska-4465831.jpg?w=1280&ssl=1)
            거주지 주변에 폐식용유를 분리 배출하는 곳이 있다면 그곳에 버리면 됩니다.

            버리는곳이 따로없다고 해서 하수구나 변기에 버려서는 안됩니다. 하수구와 변기가 막힐 수 있기 떄문입니다.

            이럴 떄는 식용유를 휴지나 종이 등에 적셔 종량제 봉투에 일반 쓰레기로 버려야 합니다.

            ### 상한 음식

            상한 음식도 음식물 쓰레기로 배출하면 됩니다.

            ### 폐의약품
            물약, 안약을 포함한 모든 약은 절대 변기나 하수구에 버리면 안됩니다.

            약의 성분들이 하수 처리과정에서 분해되지 않은 채 그대로 강과 바다로 흘러들어가 생태계를 파괴하는 일이 벌어지기 때문입니다.

            유효기간이 지나거나 먹지 않은 약이 있다면 반드시 약국이나 보건소에 가져다 주어야합니다.
            
            """

        if output in ['battery']:

            return """
            # 건전지

            <img src="http://ezlook.tel/wp-content/uploads/2020/11/%EA%B1%B4%EC%A0%84%EC%A7%80%EC%A2%85%EB%A5%98.jpg">

            건전지는 망간, 수은, 카드뮴 등의 중금속을 포함한 유해 폐기물로 환경오염을 유발하며, 심하게 훼손된 경우에는 전지액이라는 액체가 나와 피부에 닿으면 이상 증세를 일으킬 수 있다. 따라서 절대 일반 쓰레기로 버리거나 장난감이나 전자기기 속에 건전지를 넣은 채로 버리면 안 되며, 폐건전지 수거함에 따로 버려야 한다. 

            아파트의 경우에는 동마다 배치된 폐건전지 수거함에 버리면 되고, 주택에 거주하는 경우에는 모아뒀다가 주민센터에 방문해서 폐건전지 수거함에 버리면 된다.

            건전지와 비슷한 형광등 역시 유해물질인 수은이 함유되어 있어 폐형광등 전용 수거함에 버려야 한다. 형광등은 깨지는 순간 소량의 수은이 공기 중으로 방출될 수 있으므로 깨지지 않게 조심해야 하며, 깨진 형광등은 종량제 봉투에 넣어 배출해야 한다.

            건전지를 다 썼는지 확인을 위해 간단하게 잔량 테스트를 해 볼 수 있다. 건전지의 튀어나온 부분을 위로 향하게 한 후 5cm 정도의 높이에서 떨어뜨렸을 때 똑바로 선다면 건전지의 잔량이 남아 있는 상태고, 바닥에서 쓰러지거나 튕겨 오르면 다 쓴 건전지다. 다 쓴 건전지는 내부에 가스가 발생해 가벼워지기 때문이다. 

            <img src="http://www.ecolaw.co.kr/news/photo/201108/34200_10401_85.jpg">

            일반배출 시 중금속으로 아연, 이산화망간, 흑연, 염화암모늄, 니켈, 카드뮴 등이 흘러나와

            지하수 수질오염, 토양 황폐화, 대기 중 증발하여 대기오염을 일으킬 수 있습니다.

            또한 농산물, 어패류를 통해 인체로 축적되어 건강상 악영향을 끼칠 수 있습니다.

            환경과 건강을 위해 건전지가 환경에 버려지지 않도록 노력해야겠습니다.

            """

        if output in ['trash'] :
            return """
            # 일반 쓰레기

            일반 쓰레기는 일반 쓰레기 종량제 봉투에 넣어서 버리면 됩니다.

            일반쓰레기 종량제 봉투는 불투명에 가까운 반투명 흰색 봉투입니다.

            일반쓰레기 종량제 봉투 크기는 총 7종으로 2리터, 5리터, 10리터, 20리터, 50리터, 75리터, 100리터로 나뉩니다.
            
            <img src="http://image.auction.co.kr/itemimage/1a/1d/75/1a1d75dce6.jpg">
            
            """





    if drop3 == "재활용의 과정 및 제품":
        
        if output in ['paper','cardboard']: # paper임.

            return """
            # 화수 재활용 체계
            
            <img src="https://www.gimhae.go.kr/_res/depart/img/sub/p01745_img3.png"/>

            # 재활용 과정

            1. 폐지
            2. 펄프
            - 폐지를 물과 약품에 섞어 작은 섬유입자로 풀어 줌
            3. 탈묵실
            - 종이원료 속의 잉크입자 제거
            4. 원료저장소
            - 종이를 뜨는 초지과정, 탈수, 밀착, 건조 과정 및 광택과정을 거쳐 종이제조
            5. 초치기
            - 종이를 뜨는 초지과정, 탈수, 밀착, 건조 과정 및 광택과정을 거쳐 종이제조
            6. 리와인더
            - 넓은폭으로 생산된 종이를 재단하여 용도에따라 되감기 제품
            7. 제품

            # 재활용제품
            - 헌 신문지 → 신문용지
            - 인쇄지, 잡지류 → 상자류, 인쇄용지
            - 모조지, 복사지 → 골판지, 골심지 상자류
            - 헌 신문지 → 화장지, 인쇄용지

            # 재활용 효과(1톤당)
            ## 환경 오염 물질 절감 효과
            - 대기오염 74%, 수질오염 35%, 공업용수 58%

            ## 자원절약 효과
            - 30년생 나무 17그루
            - 석유 1,500ℓ(7.5드럼) 또는 전기 4,200㎾, 물 28톤(30가구 1일 사용량)
            - 쓰레기 매립지 1.7㎡
            """

        if output in ['plastic']:

            return """
            # 화수 재활용 체계
            
            <img src="https://www.gimhae.go.kr/_res/depart/img/sub/p01746_img8.png"/>

            # 재활용 과정

            <img src="https://www.gimhae.go.kr/_res/depart/img/sub/p01746_img9.png"/>
            """

        if output in ['brown-glass', 'green-glass','white-glass']:
           return """
            # 회수/재활용 체계
            
            <img src="https://www.gimhae.go.kr/_res/depart/img/sub/p01747_img1.png"/>

            # 재활용 과정

            1. 원료입고
            2. 원료평량 및 조합
            - 규사, 소다회, 석회석, 파유리를 용도 및 색조에 따라 혼합비 결정 후 혼합
            3. 용해
            - 1,500℃ 이상의 고온에서 용해
            4. 성형
            - 융용된 유리 덩어리를 성형기에서 원하는 형상으로 성형
            5. 서냉
            - 서냉로를 이용해 서서히 냉각
            6. 검사 및 가공
            7. 제품출하

            # 재활용제품

            병류의 재사용은 빈병을 회수하여 세척·소독 처리한 후 사용하는것으로 빈용기보증금제도의 정착으로 90%이상 활용되고 있습니다.
            """

        if output in ['metal']:
            return """
            # 회수/재활용 체계
            - 캔
            <img src="https://www.gimhae.go.kr/_res/depart/img/sub/p01748_img1.png"/>
            - 고철
            <img src="https://www.gimhae.go.kr/_res/depart/img/sub/p01749_img1.png"/>

            # 재활용 과정
            ## 철
            1. 수거
            - 알루미늄캔 및 이물질 분리
            2. 분리
            - 일정 부피로 압축하여 철강공자에 이송
            3. 이송
            - 용강의 냉선재 및 재강 원료로 사용
            4. 제품출하

            ## 알루미늄
            1. 수거
            - 알루미늄캔 및 이물질 분리
            2. 분리
            - 오물, 철, 기타금속 등을 제거한 후 압축
            3. 이송
            - 용해로에 500 ~ 800℃의 고열로 융용
            4. 제품출하
            - 일정 크기의 알루미늄괴로 만들어 업체에 공급

            # 재활용제품

            - 전자·공업용품 → 용강의 냉선재 및 제강원료로 사용
            - 농·어업용품 → 전열기 열판, 자동차 부품
            - 기타제품→ 각종 기계부품 생산에 사용

            # 재활용 효과(1톤당)
            ## 환경 오염물질 절감효과
            - 대기오염 85%, 수질오염 97%

            ## 자원절약 효과(알루미늄캔 1톤)
            - 보크사이트 4톤
            - 100W 전구를 4시간 켤 수있는 전기 절약
            - 에너지 절감효과 96%
            """

        if output in ['clothes','shoes']: # paper임.
            return """
            # 화수 재활용 체계
            
            1. 선별 : 금속, 플라스틱 등 이물질 제거 작업
            2. 타면 : 솜의 형태로 만드는 작업으로 3차에 걸쳐 시행하면 1차 완제품 완성
            3. 타면 : 좀 더 고운발로 섬유질화된 후 정면기에서 일정한 두께로 얇게 폄
            4. 펀칭 : 펀칭기로 들어가 PP연사로 제직한 중간원료 등과 함께 펀칭이 되어 최종제품이 됨

            # 재활용 제품
            보온덮개, 부직포, 완구류, 자동차 내장재, 이불솜, 베개 및 방석, 쿠션 등
            """

        if output in ['biological']:
            return """
            # 음식물 쓰레기

            <img src="https://www.greenkorea.org/wp-content/uploads/2022/02/11193452/recy3_1-615x616.jpg"/>
            
            전국적으로 하루에 버려지는 음식물쓰레기는 2011년 기준 13,537톤으로, 연간 약 500만톤이 발생하고 있다. 이 중 95.3%가 사료나 퇴비로 재활용되고 나머지는 매립(1.2%)하거나 소각(3.4%) 처리하고 있다. 
            음식물쓰레기는 건조 중량 기준 발열량이 높고 수분이 충분하며 유기성 물질로서 영양소도 충분하므로 과다한 염분 농도 문제 및 매운 맛과 같은 향신료 문제 등을 제거하면 퇴비나 사료로서 유용한 자원으로 재활용이 가능하다.
            """

    if output in ['battery']:

            return """
            # 건전지

            <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCEeD1%2FbtqXoaxBrcP%2FQSAwOQAKnU7T05qyNwWMp1%2Fimg.png">

            폐건전지가 수거되면 전지 종류별로 분리를 한 후

            건식, 습식, 소각, 용융 등의 공정을 거쳐 재활용에 사용되는 금속을 얻을 수 있습니다.

            주로 많이 사용되는 1차 전지인 망간, 알카라인 전지의 경우

            60%가 망간/아연 파우더로 세라믹 벽돌의 착색제로 사용되며,

            15%가 철, 아연 스크랩으로 재활용되어 철강재료로 사용됩니다.

            망간, 아연 등은 전량을 수입해야하기 때문에 재활용이 필요해 보입니다. 


            2019년 망간, 알카라인 전지의 재활용률은 25%이며

            생산자책임재활용(EPR)제도로 의무적 재활용률이 매년 증가하고 있습니다.

            """

    if output in ['trash'] :
            return """
            # 일반 쓰레기

            일반 쓰레기는 재활용이 어렵습니다.
            
            <img src="https://cdn.imweb.me/thumbnail/20220303/521a00abbe520.jpg">

            일반 쓰레기를 처리하는 방법은 크게 매립, 소각 2가지 입니다. 불에 타는 가연성 쓰레기는 소각시설에서 소각 처리 되고, 불에 타지 않는 불가연성 쓰레기는 매립장에서 매립의 과정을 거칩니다.
            
            """





# 찾아보기

def showmap(output,drop3):
    if drop3 != "찾아보기":
        return ""
    
    return"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="../css/day11.css">
</head>
<body>
    <section>
        <form action="https://search.naver.com/search.naver">
            <div class="search">
                <input type="text" name="query" value="
"""+output+" 분리수거"+"""
">
                <button type="submit">검색</button>
            </div>
        </form>
    </section>
</body>
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d25302.247717359518!2d126.72527617910151!3d37.560224!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x357c832c154138ab%3A0x8489d0feab3f5022!2z67aE66as7IiY6rGw7J6l!5e0!3m2!1sko!2skr!4v1673407486792!5m2!1sko!2skr" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
</html>
    """







# 구체적 화면 코드 

with gr.Blocks() as demo:

    gr.Markdown("# 쓰레기 분류 프로그램")
    gr.Markdown("오직 사진 한장만으로 간편하게!")
    gr.HTML("""<div style="display: inline-block;  float: right;">Made By 5 Team : 이성규, 김민정, 이승현, 이주형, 민안세</div>""")

    # 1 번탭
    with gr.Tab("Image Upload"):
        with gr.Row():

            image_input = gr.Image(label="Upload IMG")
            label_output = gr.Label(num_top_classes=3, label="Predicting percent")
        
        output = gr.Textbox(label="Predicted Label",interactive=False)

        drop3 = gr.Radio(["분리수거 방법", "재활용의 과정 및 제품", "찾아보기"], label="무엇이 궁금하신가요?")

        image_button = gr.Button("Check and Search!")
        
        with gr.Accordion("Show Information", open= False):
            info2 = gr.HTML()
            info1 = gr.Markdown("# 이미지를 넣어서 정보를 확인해주세요! ^_^")
            drop3.change(fn=showInfor, inputs=[output,drop3], outputs=info1)
            drop3.change(fn=showmap, inputs=[output,drop3], outputs=info2)

        image_button.click(predict,inputs=image_input, outputs=label_output )
        image_button.click(predict2,inputs=image_input, outputs=output )
        gr.Examples(
            examples=["./samples/sample1.png", "./samples/sample2.png", "./samples/sample3.png", "./samples/sample4.png", "./samples/sample5.png", "./samples/sample6.png", "./samples/sample7.png", "./samples/sample8.jpg", "./samples/sample9.png", "./samples/sample10.png", "./samples/sample11.png", "./samples/sample12.png"],
            inputs=image_input,
        )




    # 2번 탭
    with gr.Tab("Using WebCam"):
        with gr.Row():
            image_input = gr.Image(source="webcam", streaming=True, label="Web Cam")
            image_result = gr.Image(label="Taking IMG")
            label_output = gr.Label(num_top_classes=3, label="Predicting percent")
        
        output = gr.Textbox(label="Predicted Label",interactive=False)

        drop3 = gr.Radio(["분리수거 방법", "재활용의 과정 및 제품", "찾아보기"], label="무엇이 궁금하신가요?")

        image_button = gr.Button("Check and Search!")
        
        with gr.Accordion("Show Information", open= False):
            info2 = gr.HTML()
            info1 = gr.Markdown("# 이미지를 넣어서 정보를 확인해주세요! ^_^")
            drop3.change(fn=showInfor, inputs=[output,drop3], outputs=info1)
            drop3.change(fn=showmap, inputs=[output,drop3], outputs=info2)
        
        image_button.click(stop,inputs=image_input, outputs=image_result )
        image_button.click(predict,inputs=image_input, outputs=label_output )
        image_button.click(predict2,inputs=image_input, outputs=output )





    # 3번 탭
    with gr.Tab("Make IMG"):
        with gr.Row():
            gr.Markdown("""
            # 사진이나 가진 쓰레기가 없다구요?
            
            걱정 마세요! 원하는 쓰레기를 무료로 만들어드립니다!""")
            making = gr.Radio(["옷", "깡통", "병", "종이", "신발", "금속"], label="원하는 생성 쓰레기 선택")
        
        with gr.Row():
            maked_image = gr.Image(label="Made IMG")
            making_button = gr.Button("← Image Making! :D")


        apply_button = gr.Button("↓ 바로 적용 해보기 ↓")
        
        making_button.click(making_img,inputs=making, outputs=maked_image )

        
        with gr.Row():
            image_result = gr.Image()
            label_output = gr.Label(num_top_classes=3, label="Predicting percent")
        
        apply_button.click(stop,inputs=maked_image, outputs=image_result )

        output = gr.Textbox(label="Predicted Label",interactive=False)

        drop3 = gr.Radio(["분리수거 방법", "재활용의 과정 및 제품", "찾아보기"], label="무엇이 궁금하신가요?")

        image_button = gr.Button("Check and Search!")
        
        with gr.Accordion("Show Information", open= False):
            info2 = gr.HTML()
            info1 = gr.Markdown("# 이미지를 넣어서 정보를 확인해주세요! ^_^")
            drop3.change(fn=showInfor, inputs=[output,drop3], outputs=info1)
            drop3.change(fn=showmap, inputs=[output,drop3], outputs=info2)
        
        image_button.click(predict,inputs=image_result, outputs=label_output )
        image_button.click(predict2,inputs=image_result, outputs=output )

    with gr.Accordion("데이터로 보는 재활용", open= False):
        gr.Markdown("""

# 데이터로 보는 재활용

이 글은 쓰레기 처리 프로그램을 하기에 앞서 자원 순환 정보 시스템에 배포된 2019년 “전국 폐기물 발생 및 처리 현황” 데이터로 가정에서 배출하는 쓰레기의 종류와 양을 보고자 합니다. 사업장에서 발생하는 생활계 폐기물은 포함하지 않고, “가정 쓰레기 및 개보수 공사 등으로 인한 5톤 미만의 생활폐기물” 데이터를 사용했습니다. 시도별 하루 생활폐기물 발생 총량과 배출방식 및 처리방식에 따른 배출량을 구분하여 살펴보았습니다.

과연 우리는 매일 얼마만큼의 쓰레기를 배출하고 있을까요?

2019년에 전국의 생활폐기물 발생량은 하루 약 45,912톤입니다. 주변에서 흔히 볼 수 있는 1톤 용달 트럭 4만 5천대 분량의 쓰레기가 매일 생산된다고 볼 수 있습니다. 경기도에서만 하루 전국 생활폐기물의 20.7%를 차지하는 9,543.9톤의 쓰레기가 발생합니다. 물론 경기도에는 전국 인구의 26%가 살고 있기 때문에 경기도의 절대적인 쓰레기 발생량이 많다고 볼 수 있습니다.

# 시도별 생활폐기물 총 발생량(톤/일)
<img src="http://www.bigdata-map.kr/img/story/env/household-waste1.png"/>
그렇다면 1인당 폐기물 발생량을 기준으로 보면 어떨까요? 총 발생량을 시도별 인구수로 나누어 보면 전국의 하루 1인당 발생량은 0.86kg입니다. 0.86kg은 작게 느껴질 수 있지만, 한사람이 10일이면 8.6kg, 한달이면 약 26kg에 가까운 쓰레기를 배출하는 셈입니다.

# 시도별 1인당 생활폐기물 발생량(kg/일)

<img src="http://www.bigdata-map.kr/img/story/env/household-waste2.png"/>
특히 제주도의 1인당 하루 생활폐기물 발생량은 1.38kg으로 전국 평균치보다 약 1.6배입니다. 해당 총 발생량은 가정에서 어떠한 방식으로 폐기물이 배출되었는가를 기준으로 종량제, 재활용, 음식물 쓰레기로 세분화해 볼 수 있습니다.

# 시도별 하루 생활폐기물 중 재활용 분리 배출의 비율(%)

<img src="http://www.bigdata-map.kr/img/story/env/household-waste3.png"/>

# 시도별 생활폐기물 중 종량제 방식에 의한 혼합배출의 비율(%)

<img src="http://www.bigdata-map.kr/img/story/env/household-waste4.png"/>
1인당 배출 총량이 많았던 제주도의 경우 제주도 전체 배출량의 44.83%가 재활용 가능 자원으로 분리 배출되어, 부산광역시(47.73%), 세종특별시(47.22%) 다음으로 재활용 배출 비율이 많습니다. 반면 재활용 가능성이 낮은 종량제 방식에 의한 혼합 배출 폐기물은 제주도 전체 배출량의 31.98%로 전국 17개 시도 중 부산광역시(29.34%) 다음으로 가장 적습니다. 전국으로 보면 종량제 쓰레기는 하루 배출량의 45.68%를 차지합니다.

# 시도별 생활폐기물 중 음식물류 분리 배출의 비율(%)

<img src="http://www.bigdata-map.kr/img/story/env/household-waste5.png"/>
음식물류 폐기물로 분리 배출되는 쓰레기는 전국에서 하루 평균 13,138.8톤으로 전체 배출량의 28.62%를 차지합니다. 특히 17개 시도 중 광주광역시(40.45%), 대전광역시(35.46%), 인천광역시(34.96%)에서 하루 생활폐기물 중 음식물 쓰레기가 차지하는 비중이 높습니다.

그렇다면 음식물 폐기물은 얼마나 배출되고 있을까요?

# 시도별 음식물류 폐기물의 1인당 배출량(kg/일)

<img src="http://www.bigdata-map.kr/img/story/env/household-waste6.png"/>
1인당 배출량으로 보면, 전국에서 한 사람당 하루 0.25kg, 10일이면 2.5kg, 한달이면 약 7.5kg의 음식물류 쓰레기를 배출합니다. 전국 17개 시도 중 광주광역시(0.32kg), 제주도(0.32kg)에서 특히 많습니다. 17개 시도의 하위 지역 부분인 시/군/구 단위에서 1인당 음식물류 폐기물의 배출량을 많은 지역을 살펴보면 다음과 같습니다.


# 1인당 음식물류 폐기물(kg/일) 배출이 많은 시군구 Top 20

<img src="http://www.bigdata-map.kr/img/story/env/household-waste7.png"/>
경상북도 안동시(0.6kg), 서울시 종로구(0.58kg), 서울시 송파구(0.52kg), 경상남도 김해시(0.47kg), 서울시 서초구(0.38kg), 전라남도 여수시(0.36kg)에서 1인당 음식물류 폐기물 배출이 많습니다.

많은 우려가 제기되어 온 폐합성수지류(비닐류, 발포수지류, PET병 포함), 일반적으로 플라스틱으로 알고 있는 폐기물은 얼마나 배출되고 있을까요?

# 시도별 발생 총량 중 폐합성수지류의 비율(%)

<img src="http://www.bigdata-map.kr/img/story/env/household-waste8.png"/>


# 시도별 폐합성수지류 폐기물 1인당 배출량(kg/일)

<img src="http://www.bigdata-map.kr/img/story/env/household-waste9.png"/>

전국의 하루 폐합성수지류 폐기물의 배출량은 2,604.3톤으로 하루 총 발생량(45,912톤)의 5.67%를 차지합니다. 부산광역시의 경우 부산 전체 배출량의 11.37%로 높습니다. 한사람이 하루 0.05kg의 폐합성수지류 폐기물을 배출합니다. 전국 17개 시도 중 부산광역시(0.1kg)가 특히 많습니다. 17개 시도의 하위 지역 부분인 시/군/구 단위에서 1인당 폐합성수지류 폐기물의 배출량을 많은 지역을 살펴보면 다음과 같습니다.

# 1인당 폐합성수지류 폐기물(kg/일) 배출이 많은 시군구 Top 20

<img src="http://www.bigdata-map.kr/img/story/env/household-waste10.png"/>

서울시 동대문구(0.2kg), 부산시 서구(0.16kg), 서울시 은평구(0.16kg), 서울시 서대문구(0.12kg), 부산시 부산진구(0.11kg), 부산시 사하구(0.11kg)에서 1인당 폐합성수지류 폐기물 배출이 많습니다.

# 시도별 배출방식 별 생활폐기물의 비율(%)

<img src="http://www.bigdata-map.kr/img/story/env/household-waste11.png"/>

종합하여 보면, 하루 가정에서 배출되는 생활폐기물의 75%이상이 종량제 방식에 의한 혼합배출 (20,971.1톤)과 음식물 쓰레기(13,138.8톤)이며, 25%가 재활용 가능 자원(11,802.2톤)으로 배출됩니다. 하지만 가정에서 재활용 가능 자원으로 배출된 폐기물이 모두 재활용 처리되는 것은 아닙니다.

# 재활용으로 배출된 폐기물의 재활용 처리 비율(%)

<img src="http://www.bigdata-map.kr/img/story/env/household-waste12.png"/>

전국에서 재활용 가능 자원으로 분리 배출된 생활폐기물 중 92.59%는 재활용 처리 되었지만, 7.41%는 소각 또는 매립 처리되었습니다. 특히 17개 시도 중 광주광역시(81.24%), 전라북도(82.26%)에서 재활용 배출 폐기물의 재활용 처리 비율이 낮습니다.

# 종량제로 배출된 폐기물의 재활용 처리 비율(%)

<img src="http://www.bigdata-map.kr/img/story/env/household-waste13.png"/>

반면 전국에서 종량제 방식에 의한 혼합배출 된 폐기물의 10.62%는 재활용 폐기물로 처리되었습니다. 특히, 부산광역시(54.56%), 대구광역시(41.21%)에서 종량제 쓰레기로 배출되었지만 재활용 처리된 폐기물의 비율이 높습니다.

# 2014년~2019년 배출 총 량(톤/일)

<img src="http://www.bigdata-map.kr/img/story/env/household-waste14.png"/>

# 2014년~2019년 처리방식 별 총량(톤/일)

<img src="http://www.bigdata-map.kr/img/story/env/household-waste15.png"/>

2014년부터 2019년까지의 변화를 살펴보면 2018년에 하루 배출량이 46,749.3톤으로 가장 많고, 2019년에는 하루 약 873톤, 1.8%가 줄어들었습니다. 하지만 5년 전인 2014년 대비 2019년의 하루 배출량은 약 3,557톤, 8.4% 증가했습니다.


폐기물 처리 유형별로 나누어 보면, 2018년과 비교하여 2019년에 전체 발생량은 1.8%가 줄었지만, 재활용 처리된 폐기물이 6.8%(1,887톤) 감소한 반면, 매립 및 소각 처리한 생활폐기물은 각각 3.2%(208톤), 5.4%(672톤) 증가했습니다.


## 요약 정리

전국 시군구의 폐기물 발생 및 처리 현황 데이터로 가정에서 매일 얼마만큼의 쓰레기를 배출하는지 보았습니다.

 - 2019년에 전국에서 하루에 발생하는 가정용 생활폐기물은 45,912톤입니다.

 - 전국에서 한 사람당 발생량은 하루 0.86kg, 10일이면 8.6kg입니다.

 - 전국에서 하루 발생량의 45.68%가 종량제 방식에 의한 혼합배출(20,971.톤), 28.62%는 음식물류 폐기물(13,138.8톤), 25.71%가 재활용 가능 자원(11,802.2톤)으로 배출됩니다.
     · 전국 17개 시도 중 하루 발생량에서 재활용으로 배출된 폐기물의 비율이 높은 시도는 부산시, 세종시, 제주도가 있습니다.
     · 전국 17개 시도 중 하루 발생량에서 종량제 혼합 배출 폐기물의 비율이 높은 시도는 경상북도, 전라남도, 강원도가 있습니다.
     · 전국 17개 시도 중 하루 발생량에서 음식물류 폐기물의 비율이 높은 시도는 광주시, 대전시, 인천시가 있습니다.

 - 음식물류 폐기물은 전국에서 한 사람당 하루 0.25kg, 10일이면 2.5kg, 한달이면 약 7.5kg을 배출합니다.
     · 전국 17개 시도 중 광주광역시(0.32kg), 제주도(0.32kg)에서 특히 배출량이 많습니다.
     · 경상북도 안동시(0.6kg), 서울시 종로구(0.58kg), 서울시 송파구(0.52kg), 경상남도 김해시(0.47kg), 서울시 서초구(0.38kg), 전라남도 여수시(0.36kg)에서 많습니다.

 - 폐합성수지류(플라스틱) 폐기물은 전국에서 하루 2,604.3톤이 발생하여, 총 발생량의 5.67%를 차지합니다.
     · 한사람이 하루 0.05kg을 배출합니다. 전국 17개 시도 중 부산광역시(0.1kg)가 특히 많습니다.
     · 서울시 동대문구(0.2kg), 부산시 서구(0.16kg), 서울시 은평구(0.16kg), 서울시 서대문구(0.12kg), 부산시 부산진구(0.11kg), 부산시 사하구(0.11kg)에서 많습니다.

 - 전국에서 재활용 가능 자원으로 분리 배출된 생활폐기물 중 92.59%는 재활용 처리 되었지만, 7.41%는 소각 또는 매립 처리되었습니다. 특히 광주광역시(81.24%), 전라북도(82.26%)에서 재활용 배출 폐기물의 재활용 처리 비율이 낮습니다.

 - 반면 전국에서 종량제 방식에 의한 혼합배출 된 폐기물의 10.62%는 재활용 폐기물로 처리되었습니다. 특히, 부산광역시(54.56%), 대구광역시(41.21%)에서 종량제 쓰레기로 배출되었지만 재활용 처리된 폐기물의 비율이 높습니다.

 - 2014년부터 2019년까지의 변화를 살펴보면 2018년 대비 2019년에는 약 873톤, 1.8%가 줄어들었지만, 재활용 처리된 폐기물이 6.8%(1,887톤) 감소한 반면, 매립 및 소각 처리한 생활폐기물은 각각 3.2%(208톤), 5.4%(672톤) 증가했습니다.

 - 5년 전인 2014년과 비교하면 2019년의 하루 배출량은 약 3,557톤, 8.4%가 증가했습니다.

        """)



demo.launch(share=True)