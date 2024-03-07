import rclpy, time, ast, sys
from rdflib.term import URIRef
import numpy as np
from numpy import array
from imitrob_hri.imitrob_nlp.sentence_processor_node import SentenceProcessor
from crow_msgs.msg import SentenceProgram
from imitrob_hri.imitrob_nlp.modules.ObjectDetector import ObjectDetector
from imitrob_hri.imitrob_nlp.modules.GrammarParser import GrammarParser
from imitrob_hri.imitrob_nlp.database.Ontology import RobotProgram, RobotProgramOperator, RobotProgramOperand, RobotCustomProgram, Template
from imitrob_hri.imitrob_nlp.structures.tagging.ParsedText import ParseTreeNode
from crow_ontology.crowracle_client import CrowtologyClient
from rclpy.node import Node
from rclpy.qos import QoSReliabilityPolicy, QoSProfile

from teleop_msgs.msg import HRICommand
from time import sleep
import threading


class Talker(Node):
	def __init__(self):
		super().__init__("talkernodadwde")
		qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
		self.pub = self.create_publisher(SentenceProgram, "/nl_input", qos)
		sleep(1)
		self.sub = self.create_subscription(HRICommand, "/nlp/hri_command", self.receive_hricommand, qos)

		self.received_hricommand = False

	def receive_hricommand(self, msg):
		print("received hricommand")
		self.received_hricommand = True

def main():
	sentence = str(' '.join(sys.argv[1:]))

	rclpy.init()
	global rosnode
	rosnode = Talker()
	msg = SentenceProgram()
	msg.header.stamp = rosnode.get_clock().now().to_msg()
	msg.data = [sentence]
	rosnode.pub.publish(msg)

	executor = rclpy.executors.MultiThreadedExecutor()
	executor.add_node(rosnode)

	executor_thread = threading.Thread(target=executor.spin, daemon=True)
	executor_thread.start()

	while True:		
		time.sleep(4)
		if rosnode.received_hricommand: 
			break
		rosnode.pub.publish(msg)
		print("Didn't received")	

	print("Done, exiting")

if __name__ == '__main__':
	main()
