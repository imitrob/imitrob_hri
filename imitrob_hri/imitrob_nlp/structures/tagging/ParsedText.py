#!/usr/bin/env python
"""
Copyright (c) 2019 CIIRC, CTU in Prague
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Zdenek Kasner, Karla Štěpánová
@mail:  karla.stepanova@cvut.cz
"""
#from nlp_crow.database.Database import Database
from imitrob_hri.imitrob_nlp.structures.tagging.Tag import Tag
from collections import namedtuple
from collections.abc import Generator

from imitrob_hri.imitrob_nlp.structures.tagging.TaggedText import TaggedText, TaggedToken

import owlready2 as ow

#db = Database()

# with db.onto as onto:
class ParsedText():
    """
    The text parsed into a tree based on the sentence structure and the grammar used for parsing.
    """
    # namespace = db.onto

    #def __init__(self,language = 'en'):
        #self.lang = language

    def get_tagged_text(self) -> TaggedText:
        """
        Returns the tagged tokens without tree structure.
        """
        return self.parse_tree.flatten()

    def __str__(self):
        return str(self.get_tagged_text())


class ParseTreeNode():
    # namespace = db.onto

    def __init__(self,language = 'en'):
        self.lang = language
        self.subnodes = []

    def flatten(self) -> TaggedText:
        """
        Return the tagged text resulting from left-to-right pass through the tree.
        """
        #tagged_text = TaggedText(language = self.lang)
        tagged_text = TaggedText(language = self.lang, tokens = [], tags = [])

        for subnode in self.subnodes:
            print("============================")
            print(type(subnode))
            print(TaggedToken)
            print(type(subnode) is TaggedToken)
            print("============================")
            if type(subnode) is TaggedToken:
                tagged_text.add_tagged_token(token=subnode.token, tag=subnode.tag)
            else:
                sub_tagged_text = subnode.flatten()
                tagged_text += sub_tagged_text

        return tagged_text


# TODO check if all of these classes are saved in ontology and delete the code
# class hasOrigText(ParsedText >> str, ow.FunctionalProperty):
#     # namespace = db.onto
#     python_name = "orig_text"
#
#
# class hasParseTree(ParsedText >> db.onto.ParseTreeNode, ow.FunctionalProperty):
#     # namespace = db.onto
#     python_name = "parse_tree"
#
# class hasLabel(ParseTreeNode >> str, ow.FunctionalProperty):
#     # namespace = db.onto
#     python_name = "label"
#
# class hasSubnodes(ParseTreeNode >> ParseTreeNode):
#     # namespace = db.onto
#     python_name = "subnodes"