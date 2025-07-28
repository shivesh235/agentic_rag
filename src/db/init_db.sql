-- Create database
DROP DATABASE IF EXISTS ragdb;
CREATE DATABASE ragdb;

\c ragdb;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create tables
CREATE TABLE departments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    budget DECIMAL(15,2) NOT NULL,
    location VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE employees (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    department_id UUID REFERENCES departments(id),
    salary DECIMAL(12,2) NOT NULL,
    hire_date DATE NOT NULL,
    ssn VARCHAR(11),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    budget DECIMAL(15,2) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    department_id UUID REFERENCES departments(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE project_assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects(id),
    employee_id UUID REFERENCES employees(id),
    role VARCHAR(50) NOT NULL,
    allocation_percentage INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE financial_transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_date DATE NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    type VARCHAR(50) NOT NULL,
    description TEXT,
    department_id UUID REFERENCES departments(id),
    project_id UUID REFERENCES projects(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO departments (name, budget, location) VALUES
('Research & Development', 1500000.00, 'Boston'),
('Sales & Marketing', 1000000.00, 'New York'),
('Human Resources', 500000.00, 'Chicago'),
('Information Technology', 2000000.00, 'San Francisco'),
('Finance', 800000.00, 'New York');

-- Insert sample employees with realistic but fake data
INSERT INTO employees (first_name, last_name, email, phone, department_id, salary, hire_date, ssn)
SELECT 
    first_names[i],
    last_names[i],
    LOWER(first_names[i] || '.' || last_names[i] || '@company.com'),
    '(' || TRUNC(RANDOM() * 900 + 100)::TEXT || ') ' || 
    TRUNC(RANDOM() * 900 + 100)::TEXT || '-' || 
    TRUNC(RANDOM() * 9000 + 1000)::TEXT,
    dept.id,
    50000 + (RANDOM() * 150000)::DECIMAL(10,2),
    CURRENT_DATE - (INTERVAL '1 day' * (RANDOM() * 3650)::INTEGER),
    TRUNC(RANDOM() * 900 + 100)::TEXT || '-' || 
    TRUNC(RANDOM() * 90 + 10)::TEXT || '-' || 
    TRUNC(RANDOM() * 9000 + 1000)::TEXT
FROM 
    departments dept
    CROSS JOIN (
        SELECT UNNEST(ARRAY[
            'John', 'Sarah', 'Michael', 'Emily', 'David',
            'Jennifer', 'Robert', 'Lisa', 'William', 'Elizabeth'
        ]) AS first_names,
        UNNEST(ARRAY[
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones',
            'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez'
        ]) AS last_names,
        generate_series(1, 2) AS i
    ) names
WHERE 
    dept.id IS NOT NULL;

-- Insert sample projects
INSERT INTO projects (name, description, budget, start_date, end_date, department_id)
SELECT
    'Project ' || proj_names[i],
    'Description for Project ' || proj_names[i],
    100000 + (RANDOM() * 500000)::DECIMAL(10,2),
    CURRENT_DATE - (INTERVAL '1 day' * (RANDOM() * 365)::INTEGER),
    CURRENT_DATE + (INTERVAL '1 day' * (RANDOM() * 365)::INTEGER),
    dept.id
FROM 
    departments dept
    CROSS JOIN (
        SELECT UNNEST(ARRAY[
            'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon',
            'Omega', 'Phoenix', 'Titan', 'Atlas', 'Nova'
        ]) AS proj_names,
        generate_series(1, 2) AS i
    ) projects
WHERE 
    dept.id IS NOT NULL;

-- Insert project assignments
INSERT INTO project_assignments (project_id, employee_id, role, allocation_percentage)
SELECT 
    p.id,
    e.id,
    CASE (RANDOM() * 3)::INTEGER
        WHEN 0 THEN 'Project Manager'
        WHEN 1 THEN 'Team Lead'
        WHEN 2 THEN 'Team Member'
        ELSE 'Contributor'
    END,
    CASE (RANDOM() * 4)::INTEGER
        WHEN 0 THEN 25
        WHEN 1 THEN 50
        WHEN 2 THEN 75
        ELSE 100
    END
FROM 
    projects p
    CROSS JOIN employees e
WHERE 
    p.department_id = e.department_id
    AND RANDOM() < 0.7;

-- Insert financial transactions
INSERT INTO financial_transactions (transaction_date, amount, type, description, department_id, project_id)
SELECT
    CURRENT_DATE - (INTERVAL '1 day' * (RANDOM() * 365)::INTEGER),
    CASE 
        WHEN RANDOM() < 0.5 THEN -1 * (1000 + (RANDOM() * 50000)::DECIMAL(10,2))
        ELSE 1000 + (RANDOM() * 50000)::DECIMAL(10,2)
    END,
    CASE (RANDOM() * 4)::INTEGER
        WHEN 0 THEN 'Revenue'
        WHEN 1 THEN 'Expense'
        WHEN 2 THEN 'Investment'
        ELSE 'Transfer'
    END,
    'Transaction for ' || dept.name,
    dept.id,
    CASE WHEN RANDOM() < 0.5 THEN proj.id ELSE NULL END
FROM 
    departments dept
    LEFT JOIN projects proj ON dept.id = proj.department_id
    CROSS JOIN generate_series(1, 5) AS g
WHERE 
    dept.id IS NOT NULL;
