
import imitrob_hri.imitrob_nlp.sentence_processor_node

import rclpy



def test_run_sentence_processor_node():
    rclpy.init()
    sp = SentenceProcessor()
    sp.destroy_node()