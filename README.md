#TODO: переделать логгирование, особенно в части генерации summary по запускам с предположением об N параллельно работающих процессах.

#TODO: в логгирование добавить фсю информацию по запуску (модели конфиги и тп)

#TODO: добавить vllm инференс.

#TODO: Мэйн класс моделей нужно отрефакторить + поменять названия методов и класса на более понятные и приемлемые

#TODO: исправить датасет multiq: добавить вариант ответа, где ответ находится в нужной форме, а не вырезан из текста (LLM отвечают правильнее, чем gold answer в ряде случаев)