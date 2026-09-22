import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from vision.inference import detector as y

class BackendTests(unittest.TestCase):
    def tearDown(self):
        y.session.cache_clear()

    def fake_ort(self, providers):
        model=Mock()
        model.get_inputs.return_value=[SimpleNamespace(shape=[1,3,640,640],type='tensor(float)')]
        model.get_modelmeta.return_value=SimpleNamespace(custom_metadata_map={'task':'segment','names':"{0: 'cap', 1: 'ground'}"})
        model.get_providers.return_value=providers
        ort=Mock()
        ort.SessionOptions.return_value=SimpleNamespace()
        ort.InferenceSession.return_value=model
        return ort

    def test_cpu_default_unchanged(self):
        ort=self.fake_ort(['CPUExecutionProvider'])
        with patch.object(y.importlib,'import_module',return_value=ort) as imp:
            y.session('model','sha','package',4)
        imp.assert_called_once_with('onnxruntime')
        self.assertEqual(ort.InferenceSession.call_args.kwargs['providers'],['CPUExecutionProvider'])
        self.assertEqual(ort.SessionOptions.return_value.intra_op_num_threads,4)

    def test_ai_two_threads_and_affinity(self):
        ort=self.fake_ort(['SpaceMITExecutionProvider','CPUExecutionProvider'])
        with patch.object(y.importlib,'import_module',return_value=ort):
            y.session('model','sha','package',2,'spacemit',(8,9))
        providers=ort.InferenceSession.call_args.kwargs['providers']
        self.assertEqual(providers[0][1]['SPACEMIT_EP_INTRA_THREAD_AFFINITY'],'8;9')
        self.assertEqual(providers[0][1]['SPACEMIT_EP_INTRA_THREAD_NUM'],'2')
        self.assertEqual(ort.SessionOptions.return_value.intra_op_num_threads,1)

    def test_no_silent_cpu_fallback(self):
        ort=self.fake_ort(['CPUExecutionProvider'])
        with patch.object(y.importlib,'import_module',return_value=ort), self.assertRaises(RuntimeError):
            y.session('model','sha','package',2,'spacemit',(8,9))

    def test_bad_options(self):
        for cpus in ([0,1],[8],[8,8],[8,16],[True,9]):
            with self.assertRaises(ValueError):
                y.runtime_settings(dict(inference_threads=2,inference_provider='spacemit',inference_cpu_ids=cpus))
        self.assertEqual(y.runtime_settings({}),(1,'cpu',()))
