dataset_names = {
    'MATH500': "qq8933/MATH500",
    'AIME': "AI-MO/aimo-validation-aime"
}

models_names = {
    'deepseek-32b': "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    'deepseek-1.5b': "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
     'deepseek-7b': "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    'Qwen-math-1.5b':'Qwen/Qwen2.5-Math-1.5B'
}

def contains_keywords(text, keywords):
    return any(keyword in text.lower() for keyword in keywords)

def sentiment_analysis(text, positive_words, negative_words):
    positive_count = 0
    negative_count = 0
    last_pos_index = -1
    last_neg_index = -1

    text = text.lower() 
    for word in positive_words:
        if word in text:
            positive_count += text.count(word) 
            last_pos_index = max(last_pos_index, text.rfind(word))

    for word in negative_words:
        if word in text:
            negative_count += text.count(word) 
            last_neg_index = max(last_neg_index, text.rfind(word)) 

    if positive_count > negative_count:
        return 1  
    elif negative_count > positive_count:
        return -1  
    elif positive_count == negative_count and positive_count > 0:
        return -1 
    else:
        return 0 
