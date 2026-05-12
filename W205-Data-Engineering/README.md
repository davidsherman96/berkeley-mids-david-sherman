# W205: Data Engineering

## Course Overview
Teaches students the principles, architectures, and tools used to build scalable data systems and production-grade data pipelines. Focuses on how organizations ingest, store, process, transform, and operationalize large-scale data for analytics and machine learning applications

## Learning Objectives
- Design data pipelines by ingesting raw data from multiple sources (i.e. APIs, data warehouses) then cleaning and transforming datasets
- Manage data storage systems: learn differences and use cases for standard SQL vs. NoSQL 
- Develop an understanding of ETL vs. ELT design, scheduling and orchestration, and data dependencies

## Folder Structure

```
W205-Data-Engineering/
├── code/       # Scripts, notebooks, and source code
├── data/       # Raw and processed datasets
└── reports/    # Written reports, papers, and deliverables
```

## Final Project
Students had to access a public API of choice, pull and clean data, store it in a database, then brainstorm and code NoSQL use cases.

Working as a fictitious grocery chain, our group identified three use cases to leverage NoSQL:
- Use MongoDB to store live traffic issues known in advance to facilitate quicker delivery for employed drivers
- Use Redis, an in-memory database management system, to provide real-time truck locations and recommendations on when to refuel based on planned routes
- Use Neo4j, a graph database management system, to identify the fastest route for customers who want to take the BART (public transport) to the store
	- Connected to the BART API to pull real-time train times, estimated time between stops, then stored those in a database for us to query 


## Notes
Key libraries used: Neo4j, NumPy, Pandas, psycopg2, gmaps
