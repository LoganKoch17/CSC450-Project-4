DROP TABLE IF EXISTS Template CASCADE;

CREATE TABLE Template (
	template_id serial PRIMARY KEY,
	template_type varchar(10) NOT NULL CONSTRAINT valid_template_type CHECK (template_type IN ('upright', 'falling', 'sitting', 'lying')),
	template_characteristic varchar(10) NOT NULL CONSTRAINT valid_template_characteristic CHECK (template_characteristic IN ('edge', 'foreground')),
	image_name varchar(50),
	image bytea UNIQUE NOT NULL
);