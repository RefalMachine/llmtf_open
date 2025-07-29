#!/bin/bash

# --- КОНФИГУРАЦИЯ ---
DOCKER_NETWORK="vllm_net"
HOST_PORT=7060
CONTAINER_PORT=8888
VLLM_IMAGE="vllm/vllm-openai:v0.8.5.post1"
HEALTH_CHECK_INTERVAL_SECONDS=5
MAX_HEALTH_CHECKS=60

# Добавьте определение переменных
HOST_URL="ХОСТ МАШИНА"  # или ваш хост
JUDGE_URL="URL ДЛЯ ВАШЕГО JUDGE"  # укажите правильный URL


# --- КОНЕЦ КОНФИГУРАЦИИ ---

# Функция для остановки и удаления контейнера
ACTIVE_CONTAINER_ID=""
stop_container() {
    local container_id=$1
    
    if [ -n "$container_id" ]; then
        echo ""
        echo "--- Остановка контейнера ---"
        echo "Останавливаю и удаляю контейнер (ID: $container_id)..."
        docker kill "$container_id" > /dev/null 2>&1
        echo "Контейнер остановлен и удален."
    fi
}

# ИЗМЕНЕНИЕ 2: Функция очистки, которая будет вызываться при выходе
cleanup() {
    echo "" >&2
    echo "--- Выполняется очистка при выходе из скрипта ---" >&2
    if [ -n "$ACTIVE_CONTAINER_ID" ]; then
        echo "Обнаружен активный контейнер для остановки: $ACTIVE_CONTAINER_ID" >&2
        stop_container "$ACTIVE_CONTAINER_ID"
    else
        echo "Активных контейнеров для очистки не найдено." >&2
    fi
}

# ИЗМЕНЕНИЕ 3: Устанавливаем "ловушку" на выход из скрипта
trap cleanup EXIT

# Функция для запуска контейнера и проверки готовности
run_model() {
    local model_path=$1
    local model_name=$2
    local tp_size=$3

    local container_name="vllm_service_$$"
    
    # ИЗМЕНЕНИЕ: Направляем вывод в stderr
    echo "Запускаю VLLM контейнер '$container_name' для модели $model_name..." >&2
    
    local container_id=$(docker run -d \
        -v /home/maindev:/workdir \
        -p ${HOST_PORT}:${CONTAINER_PORT} \
        --gpus '"device=0,1"' \
        --rm \
        --name "${container_name}" \
        --network "${DOCKER_NETWORK}" \
        --entrypoint bash \
        "${VLLM_IMAGE}" \
        -c "VLLM_USE_V1=0 python3 -u -m vllm.entrypoints.openai.api_server --model ${model_path} --gpu-memory-utilization 0.95 --max_seq_len 32000 --max_model_len 32000 --port ${CONTAINER_PORT} --tensor-parallel-size ${tp_size}")
    
    if [ -z "$container_id" ]; then
        # ИЗМЕНЕНИЕ: Направляем вывод в stderr
        echo "Ошибка: не удалось запустить контейнер. Проверьте настройки Docker." >&2
        return 1
    fi
    
    # ИЗМЕНЕНИЕ: Направляем вывод в stderr
    echo "Контейнер запущен с ID: $container_id" >&2
    
    # Проверка готовности
    # ИЗМЕНЕНИЕ: Направляем вывод в stderr
    echo "" >&2
    echo "Ожидание готовности модели..." >&2
    local health_endpoint="http://${HOST_URL}:${HOST_PORT}/health"
    local is_ready=false
    
    for i in $(seq 1 $MAX_HEALTH_CHECKS); do
        local status_code=$(curl -s -o /dev/null -w "%{http_code}" "$health_endpoint" 2>/dev/null || echo "000")
        
        if [ "$status_code" -eq 200 ]; then
            # ИЗМЕНЕНИЕ: Направляем вывод в stderr
            echo "Модель готова! Статус: $status_code" >&2
            is_ready=true
            break
        else
            # ИЗМЕНЕНИЕ: Направляем вывод в stderr
            echo "Попытка $i/$MAX_HEALTH_CHECKS: сервис недоступен (статус: $status_code). Ожидание..." >&2
            sleep "$HEALTH_CHECK_INTERVAL_SECONDS"
        fi
    done
    
    if [ "$is_ready" = false ]; then
        # ИЗМЕНЕНИЕ: Направляем вывод в stderr
        echo "Ошибка: Модель не стала готова за отведенное время." >&2
        stop_container "$container_id" # stop_container тоже должен выводить в stderr
        return 1
    fi
    
    # ИЗМЕНЕНИЕ: Этот echo ОСТАВЛЯЕМ КАК ЕСТЬ. Он вернёт значение через stdout.
    echo "$container_id"
}

# Функция для запуска тестов
run_tests() {
    local model_path=$1
    local model_name=$2
    local thinking_suffix=$3
    local thinking_flag=$4
    
    echo "--- Запуск тестов для $model_name$thinking_suffix ---"
    
    python -m benchmark.llmaaj.generate_llmaaj \
        --base_url http://${HOST_URL}:${HOST_PORT} \
        --model_name_or_path $model_path \
        --model_name $model_name$thinking_suffix \
        --api_key 1 \
        --max_new_tokens 4096 \
        $thinking_flag \
        --benchmark_name ru_arena-hard-v0.1
    
    python -m benchmark.llmaaj.judge_llmaaj \
        --judge_base_url $JUDGE_URL \
        --judge_model_name_or_path /workdir/projects/models/DeepSeek-V3-0324-INT8-Channel \
        --judge_model_name deepseek \
        --judge_api_key 1 \
        --model_name $model_name$thinking_suffix \
        --benchmark_name ru_arena-hard-v0.1
}

# Основной скрипт
# ==========================================================
# НОВАЯ ВЕРСИЯ ФУНКЦИИ MAIN
# ==========================================================
main() {
    # Читаем список моделей и их параметров в цикле.
    # Каждая строка - это триплет: model_path, model_name, tp_size.
    # Просто добавьте новую строку в блок EOF, чтобы обработать новую модель.
    while read -r model_path model_name tp_size; do
        # Пропускаем пустые строки или строки, начинающиеся с # (комментарии)
        [[ -z "$model_path" || "$model_path" == \#* ]] && continue

        echo ""
        echo "========================================================"
        echo "=== Обработка модели: $model_name"
        echo "========================================================"

        container_id=$(run_model "$model_path" "$model_name" "$tp_size")
        
        # Устанавливаем ID активного контейнера для очистки в случае прерывания
        ACTIVE_CONTAINER_ID="$container_id" 
        
        if [ $? -eq 0 ] && [ -n "$container_id" ]; then
            # Запускаем тесты без "мышления"
            run_tests \
                "$model_path" \
                "$model_name" \
                "" \
                "--disable_thinking"
            
            # Запускаем тесты с "мышлением"
            run_tests \
                "$model_path" \
                "$model_name" \
                "-think" \
                ""
            
            # Останавливаем контейнер после завершения тестов для текущей модели
            stop_container "$container_id"
            sleep 10
            ACTIVE_CONTAINER_ID=""
        else
            echo "Не удалось запустить модель $model_name. Перехожу к следующей." >&2
            # Сбрасываем ID, так как запуск не удался
            ACTIVE_CONTAINER_ID=""
        fi

    done <<EOF
# Формат: model_name_or_path                     model_name        tp_size
/workdir/data/models/qwen3/QVikhr-3-4B-Instruction                              QVikhr-3-4B-Instruction        1
Qwen/Qwen3-8B                              Qwen3-8B        1
/workdir/data/models/qwen3/Qwen3-32B         Qwen3-32B         2
# /path/to/your/next/model                     My-Cool-Model-7B  1
EOF
}

# Запуск основного скрипта
main

echo ""
echo "--- Вся обработка завершена ---"
exit 0