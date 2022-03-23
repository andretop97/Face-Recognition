# Face Recognition

Projeto com fins de aprendizado sobre visão computacional, visando ser capaz de criar um programa que consiga em tempo real detectar, identificar e rastrear pessoas enquanto a mesma estiver no campo de visão da camera

# Tabela de Conteudo

<!--ts-->

- [Sobre](#face-recognition)
- [Tabela de Conteudo](#tabela-de-conteudo)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como Instalar](#como-instalar)
- [Como usar](#como-usar)
<!--te-->

# Estrutura do projeto

```shell
$ tree
├───data
│   ├───Pessoa_1
│   ├───Pessoa_2
│   └───Desconhecido
├───src
│   ├───face_detection
│   │  ├───Dlib.py
│   │  ├───DlibCnn.py
│   │  ├───face_detection.py
│   │  ├───MTCNN.py
│   │  └───OpenCv.py
│   ├───face_recognition.py
│   └───handle_data.py
├───training_models
└───main.py
```

# Como Instalar

Como primeiro passo iremos baixar o projeto:

```bash
$ git clone https://github.com/andretop97/Face-Recognition.git
```

Dentro do projeto, iremos criar nosso ambiente e executa-lo:

```bash
$ python venv -m env
$ ./envScripts/Activate
```

Agora, ja dentro do ambiente virtual, iremos instalar as dependencias do projeto:

```bash
$ pip install -r requirements.txt
```

Apos todas as dependecias instaladas, é preciso criar a pasta "data" com a seguinte estrutura:

```shell
$ tree
├───data
│   ├───Pessoa_1
│   ├───Pessoa_2
│   └───Desconhecido
```

onde voce ira substituir o nome das pastas Pessoa_N para o nome do individuo e popula-las com imagens do mesmo

# Como Usar

Apos configurado todo projeto basta ir ao terminal e copiar o seguinte comando:

```bash
$ python ./main.py
```
