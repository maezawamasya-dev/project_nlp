import requests

SERVER_URL = "http://127.0.0.1:8000"
CORPUS_FILE = "Корпус_Дмитрий.txt"

documents = []

def print_matrix(matrix):
    """Вывод матрицы с возможностью ограничения по строкам и столбцам."""
    if not matrix:
        print("Матрица пуста")
        return
    
    n_rows = input(f"Сколько строк выводить? (Enter = все {len(matrix)}): ")
    n_rows = int(n_rows) if n_rows.isdigit() else len(matrix)
    
    n_cols = input(f"Сколько столбцов выводить? (Enter = все {len(matrix[0])}): ")
    n_cols = int(n_cols) if n_cols.isdigit() else len(matrix[0])
    
    for row in matrix[:n_rows]:
        print(row[:n_cols])

def handle_nltk_function(function_name, endpoint_name):
    """Обработчик для всех NLTK функций."""
    if not documents:
        print("Сначала загрузите корпус!")
        return
    
    n_docs = input(f"Сколько документов обработать? (Enter = все {len(documents)}): ")
    n_docs = int(n_docs) if n_docs.isdigit() else len(documents)
    
    print(f"\n{function_name} для {n_docs} документов:")
    for i, doc in enumerate(documents[:n_docs]):
        try:
            response = requests.post(f"{SERVER_URL}/text_nltk/{endpoint_name}", 
                                     json={"text": doc})
            data = response.json()
            
            if "error" in data:
                print(f"Ошибка в документе {i+1}: {data['error']}")
                continue
                
            print(f"\nДокумент {i+1}: {doc[:50]}{'...' if len(doc) > 50 else ''}")
            
            if endpoint_name == "tokenize":
                print(f"Токены: {data.get('tokens', [])}")
            elif endpoint_name == "stem":
                print(f"Стеммы: {data.get('stems', [])}")
            elif endpoint_name == "lemmatize":
                print(f"Леммы: {data.get('lemmas', [])}")
            elif endpoint_name == "pos":
                print("POS-теги:")
                for word, tag in data.get('pos_tags', []):
                    print(f"  {word}: {tag}")
            elif endpoint_name == "ner":
                entities = data.get('entities', [])
                if entities:
                    print("Найденные сущности:")
                    for entity in entities:
                        print(f"  {entity['entity']} ({entity['type']})")
                else:
                    print("Сущности не найдены")
        except Exception as e:
            print(f"Ошибка при обработке документа {i+1}: {e}")

while True:
    print("\n===== NLP Клиент =====")
    print("1. Загрузить корпус")
    print("2. TF-IDF")
    print("3. Bag-of-Words")
    print("4. LSA")
    print("5. NLTK функции")
    print("0. Выход")
    choice = input("Выберите действие: ")

    if choice == "1":
        try:
            with open(CORPUS_FILE, encoding="utf-8") as f:
                documents = [line.strip() for line in f if line.strip()]
            print(f"Загружено документов: {len(documents)}")
            response = requests.post(f"{SERVER_URL}/corpus/load", json=documents)
            result = response.json()
            if "error" in result:
                print(f"Ошибка: {result['error']}")
            else:
                print(f"Corpus load: {result}")
        except FileNotFoundError:
            print(f"Файл {CORPUS_FILE} не найден!")

    elif choice == "2":
        if not documents:
            print("Сначала загрузите корпус!")
        else:
            response = requests.post(f"{SERVER_URL}/tf-idf")
            if response.status_code == 200:
                matrix = response.json()
                if isinstance(matrix, dict) and "error" in matrix:
                    print(f"Ошибка: {matrix['error']}")
                else:
                    print("TF-IDF shape:", (len(matrix), len(matrix[0]) if matrix else 0))
                    print_matrix(matrix)
            else:
                print(f"Ошибка сервера: {response.status_code}")

    elif choice == "3":
        if not documents:
            print("Сначала загрузите корпус!")
        else:
            n_docs = input("Сколько документов выводить? (Enter = все): ")
            n_docs = int(n_docs) if n_docs.isdigit() else len(documents)
            
            print(f"Bag-of-Words для {n_docs} документов:")
            for i, doc in enumerate(documents[:n_docs]):
                response = requests.get(f"{SERVER_URL}/bag-of-words", params={"text": doc})
                if response.status_code == 200:
                    vector = response.json()
                    if isinstance(vector, dict) and "error" in vector:
                        print(f"Ошибка в документе {i+1}: {vector['error']}")
                    else:
                        print(f"\nДокумент {i+1}: {doc[:50]}{'...' if len(doc) > 50 else ''}")
                        print(f"Вектор: {vector}")
                        print(f"Длина вектора: {len(vector)}")
                else:
                    print(f"Ошибка сервера для документа {i+1}: {response.status_code}")

    elif choice == "4":
        if not documents:
            print("Сначала загрузите корпус!")
        else:
            n = input("Введите количество компонентов LSA (по умолчанию 2): ")
            n = int(n) if n.isdigit() else 2
            response = requests.post(f"{SERVER_URL}/lsa", params={"n_components": n})
            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    print(f"Ошибка: {data['error']}")
                else:
                    print("Total variance:", data.get("total_variance", "N/A"))
                    print("LSA shape:", (len(data["matrix"]), len(data["matrix"][0])))
                    print_matrix(data["matrix"])
            else:
                print(f"Ошибка сервера: {response.status_code}")

    elif choice == "5":
        print("\n===== NLTK Функции =====")
        print("1. Токенизация")
        print("2. Стемминг")
        print("3. Лемматизация")
        print("4. Части речи (POS)")
        print("5. Распознавание сущностей (NER)")
        print("0. Назад")
        
        nltk_choice = input("Выберите NLTK функцию: ")
        
        if nltk_choice == "1":
            handle_nltk_function("Токенизация", "tokenize")
        elif nltk_choice == "2":
            handle_nltk_function("Стемминг", "stem")
        elif nltk_choice == "3":
            handle_nltk_function("Лемматизация", "lemmatize")
        elif nltk_choice == "4":
            handle_nltk_function("Части речи (POS)", "pos")
        elif nltk_choice == "5":
            handle_nltk_function("Распознавание сущностей (NER)", "ner")
        elif nltk_choice == "0":
            continue
        else:
            print("Неверный выбор")

    elif choice == "0":
        print("Выход")
        break

    else:
        print("Неверный выбор, попробуйте снова.")
