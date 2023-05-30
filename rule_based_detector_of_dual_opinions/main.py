import calculations as calc

def main():
    sentences, opinions_list = calc.load_opinions()
    lemmas_count = calc.unique_words_in_csv(sentences)
    lemmas = calc.prepare_lemmas(lemmas_count)
    combinations = calc.create_combinations(lemmas)
    calc.rule_based_double_quality_search(opinions_list, combinations)
    
if __name__ == "__main__":
    main()