import calculations as calc

def main():
    opinions_list = calc.load_opinions()
    term_list = calc.get_term_list()
    calc.rule_based_double_quality_search(opinions_list, term_list)
if __name__ == "__main__":
    main()