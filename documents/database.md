# Database

The database folder should be organized into the following structure under `data/database`, with each database assigned a separate folder. The utility functions are provided in path `utils/database_utils.py` and we use [DuckDB](https://duckdb.org/) for efficiency and simplicity.
```txt
- data/
    - database/
        - financial_report/
            - financial_report.json
            - financial_report.sql
            - financial_report.duckdb
        - biology_paper/
            - biology_paper.json
            - biology_paper.sql
            - biology_paper.duckdb
        - ... new database name ...
```

Under each database folder, it at least contains the following three files:
- `{database_name}.json`: the database schema file
- `{database_name}.sql`: SQL CREATE statement to build the database
- `{database_name}.duckdb`: the DuckDB file which stores the cell content
- please use consistent, lowercased and underscore splitted (pythonic) naming convention to name your database


## Database Schema File

The database schema file is structured as: (for better readability and ease of design/revision, but may not be efficient for programming and retrieving)
```json
{
    "database_name": "which should be the basename of the schema file",
    "description": "A natural language description about this database",
    "database_schema": [ // a List of table-columns dicts
        {
            "table": {
                "table_name": "readable_name_for_this_table",
                "description": "A natural language description about this table, e.g., what it contains and its functionality."
            },
            "columns": [
                {
                    "column_name": "readable_name_for_this_column",
                    "column_type": "upper_cased_data_type_string_of_DuckDB", // refer to official doc: https://duckdb.org/docs/sql/data_types/overview
                    "description": "A natural language description about this column, e.g., what is it about.",
                    "primary_key": true, // whether this column is primary key, default to false [optional]
                    "foreign_key": [
                        "foreign_table_name",
                        "foreign_column_name"
                    ], // a tuple of (table, column), where the first element is the table name, and the second element is the column name [optional]. Please ensure the existence of the referencing table-column pair
                    "default": null, // the default value which may depend on column_type [optional]
                    "nullable": true // whether column value could be null, if not provided, primary key columns are not nullable while others can [optional]
                },
                {
                    ... // other columns
                }
            ]
        },
        {

        }
    ]
}
```

- For the complete available data types, please refer to [DuckDB Data Types](https://duckdb.org/docs/sql/data_types/overview). Here are some basic types you should prioritize and use for the json field `column_type`:
    - basic types:
        - `BOOLEAN`: boolean value, true/false;
        - `INTEGER`: int4;
        - `FLOAT`: float4;
        - `DOUBLE`: float8, please use `FLOAT` with priority;
        - `DATE`: date type, containing year, month, and day, usually in the format `YYYY-MM-DD`, e.g., `2024-08-08`;
        - `TIME`: time type, containing hour, minute, and second, usually in the format `HH:MM:SS`, e.g., `22:00:00`;
        - `DATETIME`: including both `DATE` and `TIME` (alias of `TIMESTAMP`, either type is ok), usually in the format `YYYY-MM-DD HH:MM:SS`, e.g., `2024-08-08 22:00:00`;
        - `TIMESTAMPTZ`: timestamp with time zone information, usually in the format `YYYY-MM-DD HH:MM:SS±HH:MM`, e.g., `2024-08-11 14:30:00+02:00` represents August 11, 2024, at 14:30 in a time zone that is 2 hours ahead of UTC;
        - `VARCHAR`: actually, this is an alias of `STRING`, `CHAR` and `TEXT`. Please use `VARCHAR` for consistency;
        - `UUID`: only used as primary keys, can be converted or interpreted as `VARCHAR`.
    - advanced types:
        - there are some advanced and structured data types such as `ARRAY`, `LIST`, `MAP`, `STRUCT`, and `UNION`. Please refer to the [official document](https://duckdb.org/docs/sql/data_types/overview#nested--composite-types) for use cases;
        - when specifying these advanced column types, you should pay attention to the format when filling the `column_types` field, e.g., `INTEGER[3]` for `ARRAY`, `INTEGER[]` for `LIST`, and `MAP(INTEGER, VARCHAR)` for `MAP`.


## CREATE Database

Given the `.json` file, we can invoke function in `utils/database_util.py` to automatically generate the `CREATE` SQL statement and execute it to create the database, and write the corresponding `.sql` CREATE statement and the resulting `.duckdb` file into the same folder.
```pyhton
python utils/database_utils.py --database financial_report --function create_db
python utils/database_utils.py --database biology_paper --function create_db
```


## Database Content Completion

TODO: To be accomplished.
