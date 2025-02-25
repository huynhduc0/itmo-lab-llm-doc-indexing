# DocAssistant - Document Chatbot

[![Watch the video](https://img.youtube.com/vi/0R51AM7bR10/0.jpg)](https://youtu.be/0R51AM7bR10)

A Streamlit application that allows users to upload documents or enter URLs, chat with the document content using a language model, and view a dynamically generated table of contents.

## App Demo Video

[![Watch the video](https://img.youtube.com/vi/0R51AM7bR10/0.jpg)](https://www.youtube.com/watch?v=0R51AM7bR10)

## Metrics Demo Video

[![Watch the video](https://img.youtube.com/vi/LxxtUqU_giM/0.jpg)](https://youtu.be/LxxtUqU_giM)
## Description

DocAssistant is a tool designed to help users interact with documents more efficiently. Key features include:

*   **Document Upload:** Upload documents in various formats (PDF, DOCX, TXT) or provide a URL to a web page.
*   **Chat Interface:** Ask questions about the document content and receive answers powered by a language model.
*   **Dynamic Table of Contents:** Automatically generate a table of contents for uploaded documents and web pages.
*   **Session Management:** Create and manage chat sessions, allowing you to organize your interactions with different documents.
*   **Database Integration:** Stores chat sessions, messages, and document metadata in a SQLite or PostgreSQL database.

---

## Описание

DocAssistant - это приложение Streamlit, которое позволяет пользователям загружать документы или вводить URL-адреса, общаться с содержимым документа с помощью языковой модели и просматривать динамически сгенерированное содержание.

Основные характеристики:

*   **Загрузка документов:** Загружайте документы в различных форматах (PDF, DOCX, TXT) или укажите URL-адрес веб-страницы.
*   **Интерфейс чата:** Задавайте вопросы о содержании документа и получайте ответы на основе языковой модели.
*   **Динамическое содержание:** Автоматически генерируйте содержание для загруженных документов и веб-страниц.
*   **Управление сессиями:** Создавайте сеансы чата и управляйте ими, что позволяет упорядочивать взаимодействие с различными документами.
*   **Интеграция с базой данных:** Храните сеансы чата, сообщения и метаданные документов в базе данных SQLite или PostgreSQL.

---

## Key Features (English)

*   Document Upload and Processing (PDF, DOCX, URL)
*   LLM-Powered Chat Interface (Google Gemini)
*   Dynamic Table of Contents Generation (LLM-based)
*   Database Persistence for Chat Sessions, Messages, and Document Metadata
*   Right Sidebar Table of Contents Display

## Основные характеристики (Русский)

*   Загрузка и обработка документов (PDF, DOCX, URL)
*   Интерфейс чата на основе LLM (Google Gemini)
*   Динамическое создание содержания (на основе LLM)
*   Сохранение сеансов чата, сообщений и метаданных документов в базе данных
*   Отображение содержания на правой боковой панели

## Setup Instructions (English)

1.  **Clone the repository:** `git clone https://github.com/huynhduc0/itmo-lab-llm-doc-indexing`
2.  **Navigate to the directory:** `cd itmo-lab-llm-doc-indexing`
3.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
4.  **Install dependencies:** `pip install -r requirements.txt` (Create a `requirements.txt` file with all dependencies.)
5.  **Set environment variables:** Create a `.env` file in the root directory and set the following variables:

    *   `GOOGLE_API_KEY`: Your Google Gemini API key.  Get one from [https://ai.google.dev/](https://ai.google.dev/).
    *   `DB_TYPE`: (Optional) `sqlite` or `postgresql` (defaults to `sqlite`).
    *   `DB_URI`: (Optional) The database connection URI. If using SQLite, this is the path to the database file (e.g., `sqlite:///./doc_assistant.db`). If using PostgreSQL, this is the connection string (e.g., `postgresql://user:password@host:port/database`).
6.  **Apply database migrations (if using PostgreSQL):** If using PostgreSQL, you may need to set up and run database migrations (using Alembic or a similar tool) to create the tables. See the SQLAlchemy documentation for details.
7.  **Run the Streamlit app:** `streamlit run backend/main.py`

## Инструкции по установке (Русский)

1.  **Клонируйте репозиторий:** `git clone https://github.com/huynhduc0/itmo-lab-llm-doc-indexing`
2.  **Перейдите в каталог:** `cd itmo-lab-llm-doc-indexing`
3.  **Создайте виртуальную среду (рекомендуется):**
    ```bash
    python3 -м venv venv
    source venv/bin/activate  # На Linux/macOS
    venv\Scripts\activate  # На Windows
    ```
4.  **Установите зависимости:** `pip install -r requirements.txt` (Создайте файл `requirements.txt` со всеми зависимостями.)
5.  **Установите переменные среды:** Создайте файл `.env` в корневом каталоге и установите следующие переменные:

    *   `GOOGLE_API_KEY`: Ваш ключ API Google Gemini. Получите его на [https://ai.google.dev/](https://ai.google.dev/).
    *   `DB_TYPE`: (Необязательно) `sqlite` или `postgresql` (по умолчанию `sqlite`).
    *   `DB_URI`: (Необязательно) URI подключения к базе данных. При использовании SQLite это путь к файлу базы данных (например, `sqlite:///./doc_assistant.db`). При использовании PostgreSQL это строка подключения (например, `postgresql://user:password@host:port/database`).
6.  **Примените миграции базы данных (если используете PostgreSQL):** Если вы используете PostgreSQL, вам может потребоваться настроить и запустить миграции базы данных (с использованием Alembic или аналогичного инструмента) для создания таблиц. Подробности см. в документации SQLAlchemy.
7.  **Запустите приложение Streamlit:** `streamlit run backend/main.py`

## Configuration

The application is configured using environment variables. Create a `.env` file in the root directory of the project and set the following variables:

*   `GOOGLE_API_KEY`: (Required) Your Google Gemini API key. This key is used to access the Gemini language model for the chat interface and table of contents generation. You can obtain an API key from [https://ai.google.dev/](https://ai.google.dev/).
*   `DB_TYPE`: (Optional) Specifies the type of database to use. Valid values are `sqlite` or `postgresql`. If not set, the application defaults to SQLite.
*   `DB_URI`: (Optional) Specifies the database connection URI.

    *   For SQLite, this is the path to the database file (e.g., `sqlite:///./doc_assistant.db`). The database file will be created in the project directory if it doesn't exist.
    *   For PostgreSQL, this is the full connection string (e.g., `postgresql://user:password@host:port/database`). Make sure you have PostgreSQL installed and running, and that the database user has the necessary privileges.

## Конфигурация

Приложение настраивается с помощью переменных среды. Создайте файл `.env` в корневом каталоге проекта и установите следующие переменные:

*   `GOOGLE_API_KEY`: (Обязательно) Ваш ключ API Google Gemini. Этот ключ используется для доступа к языковой модели Gemini для интерфейса чата и создания содержания. Вы можете получить ключ API на [https://ai.google.dev/](https://ai.google.dev/).
*   `DB_TYPE`: (Необязательно) Указывает тип используемой базы данных. Допустимые значения: `sqlite` или `postgresql`. Если не установлено, приложение по умолчанию использует SQLite.
*   `DB_URI`: (Необязательно) Указывает URI подключения к базе данных.

    *   Для SQLite это путь к файлу базы данных (например, `sqlite:///./doc_assistant.db`). Файл базы данных будет создан в каталоге проекта, если он не существует.
    *   Для PostgreSQL это полная строка подключения (например, `postgresql://user:password@host:port/database`). Убедитесь, что у вас установлена и запущена PostgreSQL и что у пользователя базы данных есть необходимые права.

## Usage

1.  **Load a document:** Use the "New Document" section in the sidebar to upload a file (PDF, DOCX, TXT) or enter a URL to a web page.
2.  **Create a chat session:** In the "Chat Sessions" section, create a new session by providing a session name and selecting the documents you want to include in the session.
3.  **Select a chat session:** Use the dropdown menu in the sidebar to select the chat session you want to use.
4.  **Chat with the document:** In the main chat area, type your questions about the document and press Enter. The application will use the language model to generate answers based on the document content.
5.  **View the table of contents:** The table of contents for the selected document (if available) will be displayed in the second column. Click on a heading to display that section of the document
6.  **Manage Chat Sessions:** Click the “Delete” button to remove existing chat sessions

## Использование

1.  **Загрузите документ:** Используйте раздел "New Document" на боковой панели, чтобы загрузить файл (PDF, DOCX, TXT) или ввести URL-адрес веб-страницы.
2.  **Создайте сеанс чата:** В разделе "Chat Sessions" создайте новый сеанс, указав имя сеанса и выбрав документы, которые вы хотите включить в сеанс.
3.  **Выберите сеанс чата:** Используйте раскрывающееся меню на боковой панели, чтобы выбрать сеанс чата, который вы хотите использовать.
4.  **Общайтесь с документом:** В основной области чата введите свои вопросы о документе и нажмите Enter. Приложение будет использовать языковую модель для создания ответов на основе содержимого документа.
5.  **Просмотрите содержание:** Содержание для выбранного документа (если оно доступно) будет отображено во второй колонке. Щелкните заголовок, чтобы отобразить этот раздел документа.
6.  **Управление сеансами чата:** Нажмите кнопку «Удалить», чтобы удалить существующие сеансы чата.

## Metrics and Calculations

The following metrics are used to evaluate the performance of the DocAssistant:

*   **Total Questions:** 4
*   **Total Latency:** 4.72 seconds
*   **Average Latency:** 1.18 seconds
*   **Total Tokens:** 1353

### Latency per Question

*   **how to setup k8s?:** 1.31 seconds
*   **and what is k8s pod?:** 1.06 seconds
*   **how to make a k8s pod?:** 1.46 seconds
*   **oh what is kubectl?:** 0.89 seconds

### Tokens per Question

*   **how to setup k8s?:** 1.3092701435089111
*   **and what is k8s pod?:** 1.0612914562225342
*   **how to make a k8s pod?:** 1.4622581005096436
*   **oh what is kubectl?:** 0.8884408473968506

### BERTScore Metrics per Question

*   **Question: how to setup k8s?**
    *   **BertScore:** 0.80
    *   **Faithfulness:** 0.78
*   **Question: and what is k8s pod?**
    *   **BertScore:** 0.83
    *   **Faithfulness:** 0.81
*   **Question: how to make a k8s pod?**
    *   **BertScore:** 0.81
    *   **Faithfulness:** 0.78
*   **Question: oh what is kubectl?**
    *   **BertScore:** 0.82
    *   **Faithfulness:** 0.81

### Explanation

The metrics are calculated using the BERTScore, which evaluates the similarity between the generated responses and the reference answers. The faithfulness metric measures how accurately the generated responses reflect the content of the documents.

#### BERTScore (The F1 Score)

* **What it measures:** Quantifies the semantic similarity between the generated answer and the retrieved context.
* **How it's calculated:**
  * Uses pre-trained BERT embeddings to represent words and phrases.
  * Compares embeddings to find the best matches between words in the answer and context.
  * Precision (P) measures how much of the answer is found in the context.
  * Recall (R) measures how much of the context is found in the answer.
  * F1-score is the harmonic mean of Precision and Recall.
* **Interpretation:** Higher BERTScore (closer to 1.0) means high semantic similarity; lower BERTScore (closer to 0.0) means little semantic overlap.

#### Faithfulness (Approximated with BERTScore Precision)

* **What it represents:** Uses BERTScore's precision as a proxy for faithfulness.
* **How it's calculated:** Precision (P) from BERTScore.
* **Interpretation:** Higher faithfulness (close to 1.0) implies the answer is more faithful to the context; lower faithfulness (close to 0.0) implies less faithfulness.

### Объяснение

Метрики рассчитываются с использованием BERTScore, который оценивает сходство между сгенерированными ответами и эталонными ответами. Метрика faithfulness измеряет, насколько точно сгенерированные ответы отражают содержание документов.

#### BERTScore (The F1 Score)

* **Что измеряет:** Количественно оценивает семантическое сходство между сгенерированным ответом и извлеченным контекстом.
* **Как рассчитывается:**
  * Использует предварительно обученные встраивания BERT для представления слов и фраз.
  * Сравнивает встраивания, чтобы найти лучшие совпадения между словами в ответе и контексте.
  * Точность (P) измеряет, сколько из ответа найдено в контексте.
  * Полнота (R) измеряет, сколько из контекста найдено в ответе.
  * F1-score - это гармоническое среднее между точностью и полнотой.
* **Интерпретация:** Более высокий BERTScore (ближе к 1.0) означает высокое семантическое сходство; более низкий BERTScore (ближе к 0.0) означает небольшое семантическое совпадение.

#### Faithfulness (Приблизительно с точностью BERTScore)

* **Что представляет:** Использует точность BERTScore в качестве прокси для faithfulness.
* **Как рассчитывается:** Точность (P) из BERTScore.
* **Интерпретация:** Более высокая faithfulness (ближе к 1.0) означает, что ответ более верен контексту; более низкая faithfulness (ближе к 0.0) означает меньшую верность.



## Troubleshooting

*   **Database Connection Issues:**

    *   If using SQLite, make sure the path to the database file in the `DB_URI` variable is correct and that the application has write permissions to that location.
    *   If using PostgreSQL, make sure the PostgreSQL server is running, the connection string is correct, and the database user has the necessary privileges.
    *   Check the database server logs for any error messages.
*   **Gemini API Issues:**

    *   Make sure the `GOOGLE_API_KEY` variable is set correctly.
    *   Check your Gemini API usage limits and make sure you haven't exceeded them.
    *   Try a simple test to verify that the Gemini API is working correctly.
*   **ImportError:** If you encounter an `ImportError`, make sure you have installed all of the required dependencies using `pip install -r requirements.txt`.
*   **Slow Performance:**

    *   Generating table of contents with LLMs can be slow. Consider using caching mechanisms to store the generated TOCs and reuse them when possible.
    *   If you are doing network request in `def setup_stuff()`, it may take some time since Streamlit re-runs all the code.

## Решение проблем

*   **Проблемы с подключением к базе данных:**

    *   Если вы используете SQLite, убедитесь, что путь к файлу базы данных в переменной `DB_URI` указан правильно и что у приложения есть права на запись в это место.
    *   Если вы используете PostgreSQL, убедитесь, что сервер PostgreSQL запущен, строка подключения указана правильно и что у пользователя базы данных есть необходимые права.
    *   Проверьте журналы сервера базы данных на наличие сообщений об ошибках.
*   **Проблемы с API Gemini:**

    *   Убедитесь, что переменная `GOOGLE_API_KEY` установлена правильно.
    *   Проверьте лимиты использования API Gemini и убедитесь, что вы не превысили их.
    *   Попробуйте простой тест, чтобы убедиться, что API Gemini работает правильно.
*   **ImportError:** Если вы столкнулись с ошибкой `ImportError`, убедитесь, что вы установили все необходимые зависимости с помощью `pip install -r requirements.txt`.
*   **Медленная работа:**

    *   Создание содержания с помощью LLM может быть медленным. Рассмотрите возможность использования механизмов кэширования для хранения сгенерированного содержания и повторного использования его, когда это возможно.
    *   Если вы делаете сетевой запрос в `def setup_stuff()`, это может занять некоторое время, так как Streamlit перезапускает весь код.

## Contributing

We welcome contributions to DocAssistant! Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and test them thoroughly.
4.  Submit a pull request with a clear description of your changes.
5.  Follow any coding style guidelines.

## Вклад

Мы приветствуем вклад в DocAssistant! Пожалуйста, следуйте этим инструкциям:

1.  Сделайте ответвление репозитория.
2.  Создайте новую ветку для своей функции или исправления ошибки.
3.  Внесите свои изменения и тщательно протестируйте их.
4.  Отправьте запрос на включение изменений с четким описанием ваших изменений.
5.  Соблюдайте все правила стиля кода.

## License

This project is licensed under the [MIT License](LICENSE).

## Лицензия

Этот проект лицензирован в соответствии с [лицензией MIT](LICENSE).
