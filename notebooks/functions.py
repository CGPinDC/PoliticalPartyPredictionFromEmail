#Functions for addressing missing target variables
def null_status(start_nulls, df, col_nulls, category, cat_source, col_key):
    '''
    Evaluates progress of filling missing target variables

    Inputs:
    start_nulls = number, as integer, of the nulls in `df`
    df = dataframe containing nulls
    col_nulls = name of column, as string, that contains nulls to count
    category = category, as string that contains nulls within the col_nulls column 
    cat_source = name of column, as a string, that contains categories    
    col_key = name of column, as a string, that contains a unique feature of for the 
              missing variable in the col_nulls column (e.g. key, id, name, index).

    Outputs: 
    Prints current null count, number of `category` with nulls in col_nulls row, percentage of nulls classified from start.
    '''
    start_nulls = start_nulls
    null_count = df[col_nulls].isna().sum()
    print('Target Nulls: ', null_count)

    orgs = [org for org in list(df[col_key].loc[(df[cat_source]==category) & (df[col_nulls].isna())].unique())]
    print(f'{category} without target: {len(orgs)}')

    print(f'Percentage of nulls classified: {(start_nulls- null_count)/start_nulls}')