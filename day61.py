# day 61 : tool in langgraph . 





# till noe we made a chatbot , that is able to answer question , remeber , the previous convo , onbserbkility , sqllite db etc ENOTCAPABLE

#  but it not able to connect with an y of the tools . it can work like we are giving the hands. 

#  we can give ny of the tools. 

#  we zare adding 3 tools. 

#  1. calculator tool , 
# 2. internet search tool .. like what are th top news in the kathamndu 
# 3. stock tool . it will give the company real time stick tool . 


# plan of acgtion

#  1. fundamental : adding tool in th elanggraph 
#  2. addding tool in the chatbot . 


#  start -> chatnode - end 
#  user question and chatbdoe that have llm and the end 

#  requirement 
#  the chatbit need to normal convo and also theg ahnd to work . 

#  in th esimple chatnide we will have dicision making like user is doing for chatting or working thing . 

#  eG : use ased what is the acptial of nepal. ot ill chat . 
# but it alsed waht uis the l;atest news in nepal . it will do chatting and decision makinf to the action. 

#  tool node : to add in t elanggrapgh we we put all the tool in the tool node . 

#  all tool like calculator , search duck duck go and stocu price . 

# start ->chatnoe->enc
#       ->tool -> end

#  tool node is a guy who execute a tool . 

#  a pre build node 

#  acts as a bridge between teh grapgh and external tool 

# tool condition : in buldit in langhgarph . it tell what to do a tool caling or do a normal chat . 
#  its a prebuild condiitoal edge function. to dec ide . 

# from langgraph.built import toolnod r , tool_condtion
# from langchain_community.tools import DuckDuckGoSearchRun \
# form langchain_core.tool imprt tool

#  there are 2 tyoes of tool like pre build tool and custom tool . prevuild is given by langchain . and custom tool . for maingtool we use @tool decorartor. 
#  alwasy add a doc strung iwhiole making teh tool 

# make  alist of tool 

# tools = [getstock price , search tool , calcuklator ]

# llm with tool = llm.bindtools(tools)

# add a conditional edge node ("chat_node ", tool condition)

#  ut there is still an issue . 

# like if we ask normal quetion we get answer in a norllm. 

#  but when we saty stock of the apple. we get replyt in atechnial manner like in json . not how we wanted like a llm talking to us . this is baecuse we are getting ans dir4ectly fomr the tool it self 

#  a normal user wont under tsand the answ


# start ->chatnoe->enc
#       ->tool -> end

# this is the 1st problem

# and if we ask the random compkex thing like 

#  what is thestock of the apple and how much will it ocst to buy 50 sghre . 

#  we will get the wirnd or bad answer . 


#  so the current sturectire is not good. 

#  we need to modify the answer . 

#  we need to sahre the answer to the chatnode llm again . llm will lok at the answe rby the tool and then give the correct answer . 

#  it kindof the loop . 


#  we need to carte a loop betwen llm and tool . 

# start ->chatnode->end
#       ->tool abck to chatnode. 


