import rclpy, time, ast
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

from teleop_msgs.msg import HRICommand

class Talker(Node):
	def __init__(self):
		super().__init__("talkernodadwde")

		self.pub = self.create_publisher(SentenceProgram, "/nl_input", 5)

def main():
	sentence = 'Seber kostku'

	rclpy.init()
	rosnode = Talker()
	msg = SentenceProgram()
	msg.header.stamp = rosnode.get_clock().now().to_msg()
	msg.data = [sentence]
	rosnode.pub.publish(msg)
	print("Done, exiting")

if __name__ == '__main__':
	main()
