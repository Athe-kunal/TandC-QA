from TacoLLM import *
import argparse

def get_response(user_request):
    llm1_response = response_llm1(user_request)
    valid_product_names = ['Apple_Fitness+',
                'Apple_icloud',
                'Apple_Maps',
                'Apple_MediaServices',
                'Apple_Music',
                'Apple_News+',
                'Apple_Store',
                'Apple_TV+',
                'Apple_Website',
                'Google_Cloud',
                'Google_Drive',
                'Google_Gmail',
                'Google_Maps',
                'Google_Playstore',
                'Lyft',
                'Instagram',
                'Meta',
                'Reddit',
                'Snapchat',
                'Twitter',
                'Uber']
    if len(llm1_response['Product_Names'])==0:
        return f"The product is based on Question-Answering agent on top of Terms and Conditions of companies. Please ask a relevant question."
    for prod in llm1_response['Product_Names']:
        if prod not in valid_product_names:
            return f"Currently, we are supporting {', '.join(valid_product_names)}, please ask questions about the products mentioned here."
    relevant_sentences = get_relevant_sentences(llm1_response,user_request)
    # print(relevant_sentences)
    llm2_output = get_response_llm2(relevant_sentences,user_request)
    return llm2_output

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Terms and Conditions QA Agent (TACO)')
#     parser.add_argument('-u','--user',type=str,help='User query text file')
#     args = parser.parse_args()
#     file_name = args.user
#     assert file_name.split(".")[1] == "txt", "It should be a text file"
#     with open(file_name,'r') as f:
#         user_request = f.read()
#     print(get_response(user_request))