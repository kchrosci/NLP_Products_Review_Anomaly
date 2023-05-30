import calculations as calc

def main():
    opinions_list = calc.load_opinions()
    patterns = calc.get_patterns()
    calc.rule_based_double_quality_search(opinions_list)
if __name__ == "__main__":
    main()