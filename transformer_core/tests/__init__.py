"""Define tests, sanity checks, and evaluation"""
import os
tensor_path = os.path.join(os.path.dirname(__file__), 'tensors')

from .embedding_test import test_task_1, test_task_4
from .attention_test import test_task_2, test_task_3


