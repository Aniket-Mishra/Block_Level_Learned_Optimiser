import numpy as np


def test_previous_tasks(method, test_dataset, task_idx, device=None):
    previous_task_acc = {}
    for tsk_id in range(task_idx + 1):
        test_accuracy = method.evaluate(
            test_dataset[tsk_id], tsk_id, device=device
        )
        previous_task_acc[int(tsk_id)] = np.around(test_accuracy, 4)
    avg_task_acc = sum(previous_task_acc.values()) / len(previous_task_acc)
    print(
        "Average accuracy over {} tasks: {:}%".format(
            task_idx + 1, 100.0 * avg_task_acc
        )
    )
    return previous_task_acc


def test_future_tasks(method, test_dataset, task_idx, n_tasks, device=None):
    future_task_acc = {}
    for tsk_id in range(task_idx, n_tasks):
        test_accuracy = method.evaluate(
            test_dataset[tsk_id], tsk_id, device=device
        )
        future_task_acc[int(tsk_id)] = np.around(test_accuracy, 4)
    return future_task_acc
