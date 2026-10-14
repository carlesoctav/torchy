import subprocess
import sys
import time
import unittest
from pathlib import Path

from torchy.infra.colab import ColabProcess, ColabSession, ProcessGroup


class DummyColabProcess:
    def __init__(self, exit_code=0, delay=0.05):
        self.exit_code = exit_code
        self.delay = delay
        self.returncode = None
        self.terminated = False

    def wait(self, timeout=None):
        time.sleep(self.delay)
        if self.returncode is None:
            self.returncode = self.exit_code
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.terminate()


class DummySession:
    def __init__(self, session="test-sess", stop_vm=False):
        self.session = session
        self.stop_vm = stop_vm
        self.vm_stopped = False

    def terminate_vm(self):
        self.vm_stopped = True


class TestColabInfra(unittest.TestCase):
    def test_process_group_success(self):
        with ProcessGroup() as pg:
            p1 = DummyColabProcess(exit_code=0, delay=0.05)
            p2 = DummyColabProcess(exit_code=0, delay=0.08)
            pg.add("p1", p1)
            pg.add("p2", p2)
            pg.wait("p1", "p2")

        self.assertEqual(p1.returncode, 0)
        self.assertEqual(p2.returncode, 0)

    def test_process_group_with_real_popen(self):
        with ProcessGroup() as pg:
            p1 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.05); print('done')"])
            pg.add("p1", p1)
            pg.wait("p1")

        self.assertEqual(p1.returncode, 0)

    def test_process_group_fail_fast(self):
        with self.assertRaises(SystemExit):
            with ProcessGroup() as pg:
                p_good = DummyColabProcess(exit_code=0, delay=1.0)
                p_bad = DummyColabProcess(exit_code=3, delay=0.05)
                pg.add("good", p_good)
                pg.add("bad", p_bad)
                pg.wait("good")

        self.assertEqual(p_bad.returncode, 3)
        self.assertTrue(p_good.terminated)

    def test_process_group_stop_vm_toggle(self):
        sess1 = DummySession("vm-keep", stop_vm=False)
        sess2 = DummySession("vm-kill", stop_vm=True)

        with ProcessGroup() as pg:
            p1 = DummyColabProcess(exit_code=0, delay=0.05)
            p2 = DummyColabProcess(exit_code=0, delay=0.05)
            pg.add("p1", p1, session=sess1)
            pg.add("p2", p2, session=sess2)
            pg.wait("p1", "p2")

        self.assertFalse(sess1.vm_stopped)
        self.assertTrue(sess2.vm_stopped)

    def test_colab_session_stop_vm_property_and_call(self):
        import unittest.mock
        with unittest.mock.patch.object(ColabSession, "ensure_master"):
            sess = ColabSession("fake", stop_vm=False, auto_new=False)
            self.assertFalse(sess.stop_vm)
            sess.stop_vm = True
            self.assertTrue(sess.stop_vm)
            sess.terminate_vm = unittest.mock.MagicMock()
            sess.stop_vm()
            sess.terminate_vm.assert_called_once()


    def test_popen_requires_log_path(self):
        import unittest.mock
        with unittest.mock.patch.object(ColabSession, "ensure_master"):
            sess = ColabSession("fake", stop_vm=False, auto_new=False)
            with self.assertRaises(ValueError):
                sess.Popen(cmd=["echo", "1"], log_path="")

    def test_popen_dir_mapping(self):
        import unittest.mock
        with unittest.mock.patch.object(ColabSession, "ensure_master"), \
             unittest.mock.patch("subprocess.run") as mock_run:
            sess = ColabSession("fake", stop_vm=False, auto_new=False)
            # Mock SSH launch script check
            mock_run.return_value = unittest.mock.MagicMock(returncode=0)
            with unittest.mock.patch("threading.Thread"):
                proc = sess.Popen(
                    cmd=["python", "test.py"],
                    log_path="/tmp/test.log",
                    dir={"/tmp": "/content/remote_dir"},
                )
            self.assertIsNotNone(proc)
            # Find the ssh launch call
            ssh_launch_calls = [
                call.args[0] for call in mock_run.call_args_list
                if call.args and isinstance(call.args[0], list) and len(call.args[0]) >= 3 and call.args[0][0] == "ssh"
            ]
            self.assertTrue(len(ssh_launch_calls) > 0)
            launch_script = ssh_launch_calls[0][2]
            self.assertIn("cd /content/remote_dir", launch_script)


if __name__ == "__main__":
    unittest.main()


