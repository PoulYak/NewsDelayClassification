import torch
from models.news_classification import NewsClassification
import pandas as pd
from src import NewsDataset, NewsDatasetTest, get_test_dataset_dataloader, get_dataset_dataloader, add_augmented_data
from src import train_epoch, validation_epoch, fit, loss_fn
from src import test, load_model, predict
from src import save_checkpoint, load_checkpoint, compute_metrics, load_model
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from nltk.stem.snowball import SnowballStemmer

from src.model_evaluation import NewsSomeDetector

TEST_PATH = "data/raw/test_data.csv"
TRAIN_PATH = "data/raw/train_data.csv"
AUGMENTED_DATA_LABEL_1 = "data/raw/aug_1.txt"
AUGMENTED_DATA_LABEL_0 = "data/raw/aug_0.txt"

# HyperParameres
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 45
LEARNING_RATE = 1e-05
BERT_MODEL_NAME = 'DeepPavlov/rubert-base-cased'


def preprocess():
    train_df = pd.read_csv(TRAIN_PATH, index_col=0)
    test_df = pd.read_csv(TEST_PATH, index_col="id")
    train_df = add_augmented_data(train_df, AUGMENTED_DATA_LABEL_1, 1)
    train_df = add_augmented_data(train_df, AUGMENTED_DATA_LABEL_0, 0)

    train_data, val_data = train_test_split(train_df, train_size=0.8, random_state=43, shuffle=True)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    stemmer = SnowballStemmer("russian")
    train_set, training_loader = get_dataset_dataloader(train_data, tokenizer,
                                                        batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0,
                                                        stemmer=stemmer, max_len=MAX_LEN)
    val_set, val_loader = get_dataset_dataloader(val_data, tokenizer,
                                                 batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=0,
                                                 stemmer=stemmer, max_len=MAX_LEN)

    test_set, test_loader = get_test_dataset_dataloader(test_df, tokenizer,
                                                        batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=0,
                                                        stemmer=stemmer, max_len=MAX_LEN)
    return training_loader, val_loader, test_loader


def train():
    training_loader, val_loader, test_loader = preprocess()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NewsClassification(BERT_MODEL_NAME)
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    fit(model, optimizer, device, training_loader, val_loader)


def inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NewsClassification(BERT_MODEL_NAME)
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    model, optimizer, accuracy = load_checkpoint("models/checkpoints/best_200_symbols_frozen_10.ckpt", model, optimizer, device)

    t = [
        """ДУШАНБЕ, 3 декабря. /ТАСС/. Платежные системы Unistream, Western Union и Contact подключились к запущенному в Таджикистане во вторник Национальному процессинговому центру. Переводы через Contact будут доступны во второй половине дня, а через Western Union - в течение одного-двух дней, сообщили ТАСС в пресс-службе Национального банка Таджикистана."Сейчас работают российские платежные системы Contact, Unistream, с Western Union соглашение подписано, в связи с некоторыми техническими моментами заработает в течение одного-двух дней", - сказали в пресс-службе регулятора.Во вторник издание РБК сообщило, что переводы в Таджикистан приостановили Western Union, Contact и "Близко", тогда как Unistream, MoneyGram и "Золотая корона" продолжают проводить операции. При этом в понедельник в Нацбанке Таджикистана заявляли, что Unistream, Western Union и Contact уже подписали соответствующие соглашения и подключились к процессинговому центру.Национальный процессинговый центр был создан в Таджикистане для повышения эффективности системы денежных переводов и прозрачности операций, а также минимизации операционных рисков. Через центр будут осуществляться все трансграничные переводы без открытия банковского счета на территории Таджикистана, что обеспечит беспрепятственный поток переводов из России.""",
        """Московский аналог Диснейленда не успели запустить в срок. Открытие «Острова мечты» в Нагатинской пойме перенесли на начало следующего года, сообщили ТАСС в пресс-службе парка. Точная дата неизвестна, но ближайший день, на который доступны билеты, — 29 февраля. Их стоимость составляет от 2,5 до 5 тыс. руб.И это довольно высокая цена, считает вице-президент Союза ассоциаций и партнеров индустрии развлечений Анатолий Боярков: «Такой объект нужен в Москве — это круглогодичный закрытый парк, которых не так много в мире. Это тематический парк аттракционов, правда, до конца непонятно, чему он посвящен, потому что там большой набор различных зон, и надо посмотреть, как они будут сочетаться. Все зависит от того, как будет оказываться услуга — пока это проблема. Успех любого тематического парка зависит от того, как часто люди будут возвращаться. Ценовая политика очень важна: если повысить стоимость билетов, а ожидания не оправдаются, то это ни к чему хорошему не приведет. Мне кажется, цены на билеты в “Московский Диснейленд” немного завышены».«Остров мечты» должен стать крупнейшим крытым парком развлечений Европы. Его площадь составит 300 тыс. кв метров. Во вторую очередь войдут отель, концертный зал и яхтенная школа. Парк строят группа компаний «Регионы», которой владеет семья депутата Госдумы Зелимхана Муцоева. В проект инвестирует ВТБ: банк выделил кредитную линию на 37 млрд руб. Но общие затраты превысят $1,5 млрд, ранее говорил член совета директоров «Регионов» Амиран Муцоев. Окупится ли «Остров мечты»? Первый вице-президент Российской ассоциации парков и производителей аттракционов Игорь Родионов считает, что на это уйдут долгие годы: «Мне кажется для этого потребуется около десяти лет. До людей еще надо донести, что это такое. У зарубежного посетителя не возникает вопроса, за что они отдают деньги. У них Диснейленд существует уже 60 лет, и они прекрасно понимают, за что они платят. В Москве это первый парк подобного формата. Этот проект уникальный и нужный, потому что дефицит развлекательных комплексов очевиден, российская столица в этом плане находится в списке отстающих.После расчистки последних площадок в Парке Горького и на ВДНХ у нас в последние годы фактически нет аттракционов для взрослой аудитории, хотя для детей еще что-то есть. Не говоря уже о каких-то тематических аттракционах, которые действительно посвящены чему-то и отправляют тебя в невероятное путешествие. Поэтому такой объект, безусловно, нужен и в России, и в Москве — он сможет обеспечить и свой естественный трафик, и туристический. А для таких объектов, как тематический парк, это обязательный элемент, иначе просто он не окупится никогда».В Москве группа компаний «Регионы» занимается еще одним проектом в сфере развлечений — она строит колесо обозрения на ВДНХ, которое должно стать крупнейшим в Европе. Против работ выступают местные жители — они недовольны близостью колеса к домам.Кирилл Абакумов""",
        '''Графики обслуживания внутриквартирного и внутридомового газового оборудования,"В соответствии с п.42 Постановления Правительства РФ от 14.05.2013г. №410, собственники жилых помещений в многоквартирном доме и частных домовладений обязаны обеспечивать доступ представителей специализированной организации к внутридомовому и (или) внутриквартирному газовому оборудованию для проведения работ (оказания услуг) по техническому обслуживанию и ремонту указанного оборудования (ТО ВДГО). В целях обеспечения безопасности жизни и здоровья граждан, предотвращения возможных аварийных ситуаций при эксплуатации внутриквартирного и внутридомового газового оборудования собственникам квартир и домовладений необходимо обеспечить доступ для планового проведения ТО ВДГО специалистами АО «Шадринскмежрайгаз». О дате проведения работ по ТО ВДГО можно узнать в газете «Ваша выгода» № 98 от 11.12.2018 г, на сайте АО «Шадринскмежрайгаз» www.kurgangazcom.ru в разделе «Графики обслуживания ВДГО», по телефону 04 (моб. 104), а также на официальном сайте Администрации города Шадринска. В случае невозможности нахождения дома во время производства работ, необходимо обратиться в АО «Шадринскмежрайгаз» по тел. 04 (моб. 104) и согласовать удобный день и время выезда специалистов для выполнения работ по техническому обслуживанию газопроводов и газового оборудования. Кроме того, с информацией о недопускниках можно ознакомиться на официальном сайте АО «Шадринскмежрайгаз» www.kurgangazcom.ru в разделе «Недопуски ТО ВДГО». ''',
        '''В 2009 году компания «Запад-Строй» начала возведение торгово-развлекательного центра, финансируемого дольщиками, но строительство замерло на стадии возведения каркаса здания.'''
    ]
    news_df = pd.DataFrame({"text": t, 'title': ['' for i in range(len(t))]})

    detector = NewsSomeDetector(news_df, tokenizer, model, device=device, max_len=MAX_LEN)
    detector.run()
    outputs = detector.predict()
    print(outputs)


if __name__ == "__main__":
    inference()
