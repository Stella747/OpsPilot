
import json


# Load the application logs from the JSON file
with open("data/application_logs.json", "r") as file:
    application_logs = json.load(file)



# Select the service we want to investigate
target_service = "Payment API"


# Keep only logs belonging to that service
relevant_logs = []

for log in application_logs:
    if log["service"] == target_service:
        relevant_logs.append(log)



# Separate errors and warnings from normal messages
important_logs = []

for log in relevant_logs:
    if log["level"] == "ERROR" or log["level"] == "WARNING":
        important_logs.append(log)


# Display important logs
print("Important logs requiring attention:")
print(f"Total important logs: {len(important_logs)}")
print()

for log in important_logs:
    print(f"Log ID: {log['log_id']}")
    print(f"Level: {log['level']}")
    print(f"Message: {log['message']}")
    print("-" * 50)
