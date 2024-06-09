# llmtf_open
llmtf_open - это фреймворк для быстрой и удобной оценки качества как базовых (foundation), так и инструктивных LLM.

## Ключевые особенности:
1) Помимо стандартного инфера на hf реализовано использование vllm (--vllm ключ для evaluate_model.py), благодаря чему существенно ускоряется процесс оценки качества.
2) Модели можно оценивать как на генеративных задачах, так и с оценкой вероятностей следующего токена. (в планах добавления расчета перплексии как отдельного режима)
3) Поддержка чат темплейтов инстракт моделей (нужно задать соответствующий конфиг).
4) Возможность достаточно просто добавлять свои датасеты для оценки.
5) Помимо итогового скора логгируется весь input и output для каждого сэмпла.

## Примеры работы с фреймворком:
TODO: добавить examples.

## Из датасетов на данный момент:
1) darumeru бенчмарк (https://huggingface.co/datasets/RefalMachine/darumeru). Создан на основе validation и train сплитов ['rcb', 'use', 'rwsd', 'parus', 'rutie', 'multiq', 'rummlu', 'ruworldtree', 'ruopenbookqa'] датасетов из MERA с некоторыми изменениями (пополнение multiq варивнтов ответов, при оценки USE прощаются лишние пробелы между вариантами ответов: .strip(), везде добавлена часть с "Ответ:")
2) mmlu_ru и mmlu_en из NLPCoreTeam/mmlu_ru. Получаемые скоры сопостовимы с получаемыми через https://github.com/NLP-Core-Team/mmlu_ru с небольшим отличием. (конфиг для полного воспроизведения будет добавлен позже). 

## Некоторые особенности/условности:
1) По умолчанию зафиксирован xformers бэкенд для vllm в качестве flash attention, так как FA2 с батчем > 1 выдает некорректные результаты в logprobs.
2) Так как darumeru бенч создан на основе валидации/трейна, что-то могло попасть в обучение каким-либо моделям. Это не новость, просто нужно иметь в виду.
3) В случае желания оценить lora адаптеры: 1. если были modules_to_save то vllm не позволяет такое оценивать -> используйте без --vllm флага. 2. В общем не проводилось пока тестирование этого случая.
4) Квантизация не тестировалась и почти наверняка могут быть проблемы с этим.
5) Для более стабильных результатов batch_size=1
6) В случае использования моделей с sliding_window и vllm могут быть особенности. Внимательно читайти логи запуска по этому поводу.
7) Длина max_len, задаваемая при запуске, может быть изменена для оценки задач, в случае, если max_len + Task.max_new_tokens > max_model_len. Читайте логи.
8) darumeru/rutie имеет смысл сравнивать при одинаковых значениях max_len!!! В ином случае, чем max_len меньше, тем качество будет выше, так как длина context в промпте фиксируется и старые сообщения диалога отбрасываются.
9) Параметры генерации фиксируются при инициализации модели. Захардкожено в коде на данный момент. Их можно переписать передав на функцию оценки generation_config с нужными параметрами, но этот случай пока что особо не тестировался.
10) Task может иметь кастомные stop_tokens для генерации. В случае vllm работает и для последовательности токенов, в случае hf берется всегда только первый токен из последовательности (будет изменено в будущем)