import javalang
import os
import numpy as np
import pandas as pd
import re

methods_name = []
class_feature = {}
features = {}
current_class = None

def RFC_finder(methods_in_class):
    public_method = called_methods = 0
    if "public" in methods_in_class.modifiers:
        public_method += 1
    for path1, node1 in methods_in_class:
        if type(node1) is javalang.tree.MethodInvocation:
            called_methods += 1
    RFC = public_method + called_methods
    return RFC

for paths, dirs, files in os.walk("resources/defects4j-checkout-closure-1f/src/com/google/javascript/jscomp"):
    for name in files:
        if '.java' in name:
            fullpath = os.path.join(paths, name)
            f = open(fullpath, 'r')
            f_read = f.read()
            tree = javalang.parse.parse(f_read)
            for path, node in tree:
                file_name = name.replace('.java', '')
                if type(node) is javalang.tree.ClassDeclaration and node.name == file_name:
                    current_class = node.name
                    class_feature = {}

                    # Class metrics
                    class_feature["MTH"] = len(node.methods)
                    class_feature["FLD"] = len(node.fields)
                    class_feature["RFC"] = 0

                    for methods_in_class in node.methods:
                        class_feature["RFC"] += RFC_finder(methods_in_class)

                    if node.implements is not None:
                        class_feature["INT"] = len(node.implements)
                    elif node.implements is None:
                        class_feature["INT"] = 0

                    # Method metrics
                    current_statement = max_statement = 0
                    current_CPX = max_CPX = 0
                    current_return = max_return = 0
                    current_throws = max_throws = 0
                    total_statement = 0

                    methods_name = []
                    for method in node.methods:
                        if method.throws is not None:
                            current_throws = len(method.throws)
                        if method.name is not None and method.name != "":
                            methods_name.append(method.name) #NML

                        for path2, node2 in method:
                            if type(node2).__base__ is javalang.tree.Statement and type(node2) is not javalang.tree.BlockStatement:
                                current_statement += 1
                                total_statement += 1

                            if type(node2) is javalang.tree.IfStatement or \
                                    type(node2) is javalang.tree.ForStatement or \
                                    type(node2) is javalang.tree.WhileStatement:
                                current_CPX += 1

                            if type(node2) is javalang.tree.ReturnStatement:
                                current_return += 1

                        if current_statement > max_statement:
                            max_statement = current_statement
                        if current_CPX > max_CPX:
                            max_CPX = current_CPX
                        if current_return > max_return:
                            max_return = current_return
                        if current_throws > max_throws:
                            max_throws = current_throws

                        current_statement = 0
                        current_CPX = 0
                        current_return = 0
                        current_throws = 0

                    class_feature["SZ"] = max_statement
                    class_feature["CPX"] = max_CPX
                    class_feature["EX"] = max_throws
                    class_feature["RET"] = max_return

                    # NLP metrics
                    word_count = 0
                    doc = 0
                    for path3, node3 in node:
                        if hasattr(node3, 'documentation') and node3.documentation is not None:
                            if isinstance(node3.documentation, str):
                                doc += 1
                                word_count += len(re.findall(r'\w+', node3.documentation))
                            elif type(node3.documentation) == list:
                                doc += len(node3.documentation)
                                for comment in node3.documentation:
                                    word_count += len(re.findall(r'\w+', comment))

                    class_feature["BCM"] = doc
                    if methods_name != []:
                        class_feature["NML"] = np.average([len(i) for i in methods_name])
                    class_feature["WRD"] = word_count

                    if class_feature["SZ"] != 0:
                        class_feature["DCM"] = class_feature["WRD"] / total_statement

                    features[current_class] = class_feature

from_dict = pd.DataFrame.from_dict(features, orient='index')
df = from_dict.reset_index().rename(columns={"index": "class"}).fillna(0)
print(df.to_string())
df.to_csv('feature_vector_file.csv')

