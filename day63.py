# day 63: rag in the langgaroh 

#  till now we have make a chatbot now wqe will ake a rag chatbot . 

# we can upload the pdf file and we can do converstation

# multi utility chatbot 


#  it can use tool , mcp , amnd rag a,dnoral  chatbot


#  plan of action :

# 1. quick recap  of rag 

# 2. separate new code rag in laggraph . 

# 3. inregrate in existing chatbot 

#  why is rag importnat . otdated knweledge . 

#  eveeyr llm have knowlwedge cutof date 

# the taruingin is completed ina somee partivualr date . 

#  by defination like a llm get ready of 31 ausgust 2026 . but todasy we ask it wont be anble to answer . 

#  but you might think chatgot reply the latest answer . but behinfd it call t he web search and than answer 

# 2nd biggest prblem solve is privacy : like gopt csn answer anything but if we ask us about the personal questio like oir slary . than it wonr be able to answwr ,. 

#  3rd is halluilicatio : gives false informatopon .with confidence . it is stiil a big probelm . 

#  rag helps you with the hallicualtion problem . 

#  the biggest uswe case is tthese /. 


#  how rag work . 

#  rag work in the hsimple pricinple like inciontext el;arnig like if we give cintext to the llm . llm get ts the queruy and the context . 
#  we give promt to llm . llm have the parametroc knowledge ., 

#  liek we have the expense sheet , 
#  we ask abouyrt teh llm but llm parametruc knowlege doesnit have itx knowlegeese. 


# what we do is we give the expexe sheet and give the whole expense sheet 

# tye ony priblem is we cannot give te context evey time . becauer it have context sioze . like if the folder have 100 books  now the information will cronss the contetx sizer . 

#  2nd eg : we have the code base . we only gie ve the cintext every time we ask a quwation. ww  only give context kthat are required . 

#  context fuilterting is the most imp part . 

# incontext learnig , context filtering 


# pdf -> chunk-> embedding if each chunk (comnvert int o vector 0) that is it capture the semtic meaning idf each chunk -> we get each chink emddding -> save in the vectore store eg fiass , chroma . 

# user - > query -> retriver (convert the question into the embedding )-> compare query emedding woth the vector store like it search for semantic searcjh. like chunk 1, 4 amd 5 his sementoc search -> in evestore we store the embedding and its real data as well -> promt (query + context that is all 3 chunk we get ) -> llm answer 

# rag in langgraph 

# tom ijmplemet rag there are multiple ways . 

#  we will deifne it as a tool . its a best way fo doing it . 
#  alwasys try to do rag as a tool . 


