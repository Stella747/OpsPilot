from database import engine, Base
from models import IncidentDB


Base.metadata.create_all(bind=engine)

print("Database and tables created successfully!")