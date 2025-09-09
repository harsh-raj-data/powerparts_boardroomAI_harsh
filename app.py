import os
import json
import uuid
import threading
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import mysql.connector
from mysql.connector import Error
from openai import OpenAI
import pandas as pd
import logging
from pathlib import Path
import time
import re
import faiss
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from datetime import datetime, date
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from decimal import Decimal
import hashlib
import re
from dotenv import load_dotenv
import requests

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="PowerParts RAG System", version="1.0")

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Templates configuration
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# In-memory storage for query results
query_cache = {}
query_results = {}
query_lock = threading.Lock()

# Database Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    'use_pure': True
}

# History Database Configuration
HISTORY_DB_CONFIG = {
    "host": os.getenv("HISTORY_DB_HOST"),
    "port": int(os.getenv("HISTORY_DB_PORT", 3306)),
    "user": os.getenv("HISTORY_DB_USER"),
    "password": os.getenv("HISTORY_DB_PASSWORD"),
    "database": os.getenv("HISTORY_DB_NAME"),
    'use_pure': True
}

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4-turbo-preview"
OLLAMA_API_URL = os.getenv("API_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
API_KEY = os.getenv("API_KEY", "super-secret-key")

# Schema definition
TABLE_SCHEMAS = {
  "engine_sales_records": {
    "description": "Stores detailed information about engine or generator sets sold, including technical specifications like engine_serial_number, model_number, engine_family, original_equipment_manufacturer, warranty status, ownership, location, and service history. Useful for population tracking number of engines sold, service planning, and warranty monitoring.",
    "columns": {
      "engine_serial_number": "Unique serial number of the engine.",
      "model_number": "Model number of the genset or engine.",
      "engine_family": "Classification group or family of the engine.",
      "alternator_make": "Brand or manufacturer of the alternator.",
      "original_equipment_manufacturer": "Original Equipment Manufacturer of the genset.",
      "rating_in_kilowatt_amperes": "Power rating of the genset in kilovolt-amperes (kVA).",
      "captiva_invoice_date": "Date when the genset was invoiced or Sold by Captiva.",
      "in_service_date": "Date when the genset was sold and became operational.",
      "inside_warranty": "Indicates whether the genset is within the warranty period (NON-WARRANTY / WARRANTY).",
      "customer_name": "Name of the end customer or organization using the genset.",
      "customer_segment": "Business segment or vertical the customer belongs to.",
      "contact_person_name": "Primary point of contact at the customer's organization.",
      "contact_person_number": "Phone number of the contact person.",
      "customer_address": "Physical address where the genset is installed.",
      "customer_e_mail_id": "Email ID of the customer contact.",
      "customer_city": "City where the genset is located.",
      "customer_state": "State where the genset is located.",
      "customer_region": "Customer region where the genset is located.",
      "branch": "Branch office responsible for the customer or site.",
      "number_of_running_hours": "Total running hours logged by the genset.",
      "last_routine_service_date": "Date of the last scheduled routine service.",
      "last_routine_service_running_hours": "Running hours recorded during the last routine service.",
      "under_amc_need_to_be_updated": "Indicates whether the genset is covered under AMC and needs updating (Yes/No/Blank).",
      "engine_prefix_number": "engine_prifix_number uniquely identifies engine categories and enables relational joins with tables such as parts_number_details to map compatible components."
    }
  },
  "sales_invoice_details": {
    "description": "Captures detailed information about engine and parts sales, services also including customer details, tax components, itemized billing, shipping information, and invoice metadata. Useful for financial tracking, taxation compliance, and sales analytics of the company with different financial year.",
    "columns": {
      "warehouse": "Location or warehouse from where goods were dispatched or billed.",
      "financial_year": "The financial year during which the invoice was generated (e.g., 2024-25).",
      "customer_segment": "Business segment or vertical the customer belongs to.",
      "invoice_number": "Unique identifier for the sales invoice.",
      "invoice_date": "Date when the sales invoice was generated.",
      "invoice_customer_post_office_number": "Reference number from customer's post office or internal dispatch system.",
      "invoice_customer_post_office_date": "Date associated with the customer's post office number.",
      "invoice_customer_name": "Name of the customer to whom the invoice is issued.",
      "invoice_billing_address": "Billing address of the customer as mentioned in the invoice.",
      "invoice_billing_gstin_number": "Customer’s GST Identification Number used for billing.",
      "invoice_shipping_address": "Shipping or delivery address for the invoice.",
      "hsn_code": "Harmonized System of Nomenclature code used for identifying the goods.",
      "part_number": "Internal or OEM part number of the product being sold.",
      "part_name": "Descriptive name of the part or item sold.",
      "unit_price": "Price per unit of the item before taxes and discounts.",
      "quantity": "Number of units sold or billed.",
      "cgst_in_percent": "Percentage of Central GST applicable to the item.",
      "sgst_in_percent": "Percentage of State GST applicable to the item.",
      "igst_in_percent": "Percentage of Integrated GST applicable for inter-state transactions.",
      "cgst_amount": "Monetary amount of CGST charged in the invoice.",
      "sgst_amount": "Monetary amount of SGST charged in the invoice.",
      "igst_amount": "Monetary amount of IGST charged in the invoice.",
      "discount_in_percent": "Percentage of discount applied to the item price.",
      "discount_amount": "Total discount amount applied before tax calculations.",
      "taxable_amount": "Net amount subject to tax after discount.",
      "net_amount": "Final billed amount including all applicable taxes.",
      "delivery_invoice_number": "Reference number of the delivery invoice or challan.",
      "invoice_e_way_bill_number": "E-Way Bill number used for transport compliance.",
      "invoice_e_way_bill_date": "Date on which the E-Way Bill was generated.",
      "invoice_shipping_city": "City where goods were shipped to as per the invoice.",
      "invoice_type": "Type of invoice (e.g., Engine and Parts Sales, AMC, Stock Transfer etc).",
      "invoice_remarks_note": "Any additional comments, notes, or internal references included with the invoice."
    }
  },
  "dead_inventory_merge": {
    "description": "Contains data related to non-moving or obsolete inventory items across different regions. Includes part identification, stock valuation, bin location, and additional remarks to support inventory management and disposal planning.",
    "columns": {
      "part_number": "Unique identifier for the inventory item or part number.",
      "part_name": "Descriptive name of the part or component.",
      "stock_quantity": "Number of quantity of the item currently in stock.",
      "landed_price": "Final cost of the item including purchase price, freight, and taxes.",
      "total_price": "Total value of the stock, calculated as stock quantity multiplied by landed price.",
      "bin_number": "Storage location or bin code within the warehouse or inventory system.",
      "sub_category": "Secondary classification of the part for inventory grouping or analysis.",
      "market_value": "Estimated current market value of the part.",
      "remarks": "Additional notes, status, or disposition instructions for the inventory item.",
      "region_name": "Region where the customer or contract is located i.e. Gurugram and Kolkata."
    }
  },
  "annual_maintainance_contracts_merge": {
    "description": "Maintains detailed records of annual maintenance contracts (AMC) including customer details, contract status, equipment specifications, service visits, and financial data. Used for tracking AMC performance, renewals, AMC expiry and support management.",
    "columns": {
      "region_name": "Region where the customer or contract is located i.e. Gurugram and Kolkata.",
      "amc_status": "Current status of the AMC (e.g., Active, Closed, Under Renewal).",
      "customer_name": "Name of the customer or organization holding the AMC.",
      "customer_address": "Physical address of the customer site or office.",
      "customer_location": "Specific location or city of the customer.",
      "customer_state": "State in which the customer is located.",
      "supported_by": "Name of the support team or engineer managing the AMC of the customer.",
      "customer_contact_person_name": "Primary contact person for AMC-related communications.",
      "customer_contact_person_mobile_number": "Mobile phone number of the contact person.",
      "amc_or_oem": "Indicates if the contract is AMC or OEM support.",
      "original_equipment_manufacturer": "Original Equipment Manufacturer or OEM related to the AMC.",
      "make_by": "Brand or manufacturer of the equipment covered under the AMC.",
      "dg_set_power_rating": "Power rating of the diesel generator set (in kVA or kW).",
      "number_of_sets_sold": "Number of generator sets sold under the contract to a single customer.",
      "engine_family": "Family or classification of the engine model.",
      "engine_model_number": "Specific model number of the engine.",
      "engine_serial_number": "Unique serial number of the engine.",
      "perkins_dispatched_date": "Date when Perkins dispatched the equipment.",
      "type_of_visit": "Type of AMC visit (e.g., Routine, Breakdown, Inspection).",
      "amc_period_from": "Start date of the AMC contract period.",
      "amc_expiry_date": "Tracking the expiry date of AMC contracts is essential for managing customer support eligibility and ensuring timely renewals.",
      "order_value_including_gst": "Total order value including GST tax.",
      "monthly_cost": "Monthly AMC cost or charge billed to the customer.",
      "engine_prefix_number": "engine_prifix_number uniquely identifies engine categories and enables relational joins with tables such as parts_number_details to map compatible components."
    }
  },
  "parts_number_details": {
    "description": "Contains data related to recommended stocking parts for engines, including part identification, and associated engine prefixes. Useful for tracking part compatibility with our customers using the same engine identified by engine_prefix_number.",
    "columns": {
      "node_name": "Group or reference list name representing the recommended stocking list for specific engine builds.",
      "part_name": "Name or description of the bigger component or part.",
      "latest_part_number": "Most recent or updated part number version, if available.",
      "part_number": "Original or standard identifier for the part.",
      "engine_prefix_number": "engine_prifix_number uniquely identifies engine categories and enables relational joins with tables such as engine_sales_records or annual_maintainance_contracts_merge to map compatible components.",
    }
  },
  "warehouse_inventory_status": {
    "description": "The table presents detailed information on the inventory status at two warehouse or region locations: Gurugram and Kolkata. It includes various parts along with their part numbers, names, available quantities, and total value (derived by multiplying the weighted landing amount with the total quantity). All items are classified under the (part_type) category (such as Parts, DG Set, Engine, etc.). The weighted landing amount represents the cost of each individual part.",

    "columns": {
        "warehouse": "Name of the warehouse or location where the part is stored (e.g., GURUGRAM, KOLKATA). Also known as region.",
        "part_number": "Unique identifier or part number assigned to each part for tracking and reference.",
        "part_name": "Descriptive name of the part or item.",
        "total_quantity": "Quantity of the part currently available in stock.",
        "part_type": "Category or type of item (e.g., Parts, DG Set, Engine).",
        "weighted_landing_amount":"Cost per unit of the part.",
        "bin_number": "Bin or shelf location within the warehouse where the part is stored.",
        "total_value": "Total inventory value of the part, calculated as total_quantity × weighted_landing_amount."}
  }
}


class UserManager:
    """Handles user registration, authentication, and management."""
    
    def __init__(self):
        self.connection = None
        self.setup_connection()
    
    def setup_connection(self):
        """Establish connection to history database."""
        try:
            self.connection = mysql.connector.connect(**HISTORY_DB_CONFIG)
            if self.connection.is_connected():
                logger.info("Connected to user database")
                self._create_tables_if_not_exist()
        except Error as e:
            logger.error(f"Error connecting to user database: {e}")
    
    def _create_tables_if_not_exist(self):
        """Create required tables if they don't exist."""
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id VARCHAR(36) PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP NULL
        )
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(create_users_table)
            self.connection.commit()
            cursor.close()
        except Error as e:
            logger.error(f"Error creating users table: {e}")
    
    def register_user(self, email: str, name: str, password: str) -> bool:
        """Register a new user with email and password."""
        if not email or not password or not name:
            return False
        
        try:
            # Simple password hashing (in production, use proper hashing like bcrypt)
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            cursor = self.connection.cursor()
            insert_query = """
            INSERT INTO users (id, email, name, password_hash)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                str(uuid.uuid4()),
                email,
                name,
                password_hash
            ))
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            logger.error(f"Error registering user: {e}")
            return False
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate a user and return user data if successful."""
        if not email or not password:
            return None
        
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            cursor = self.connection.cursor(dictionary=True)
            select_query = """
            SELECT id, email, name FROM users
            WHERE email = %s AND password_hash = %s
            """
            cursor.execute(select_query, (email, password_hash))
            user = cursor.fetchone()
            cursor.close()
            
            if user:
                # Update last login time
                update_query = """
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE id = %s
                """
                cursor = self.connection.cursor()
                cursor.execute(update_query, (user['id'],))
                self.connection.commit()
                cursor.close()
                
                return user
        except Error as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    def close_connection(self):
        """Close the database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("User database connection closed")

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

class VectorStoreManager:
    """Handles loading and querying the vector store for data context."""
    
    def __init__(self):
        # Initialize embeddings FIRST
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vector_store = None
        self.text_chunks = None
    
    def load_vector_store(self, index_path: str, chunks_path: str):
        """Load the FAISS vector store and text chunks."""
        try:
            # Verify paths exist
            if not Path(index_path).exists():
                logger.error(f"FAISS index file not found at {index_path}")
                return False
            if not Path(chunks_path).exists():
                logger.error(f"Text chunks file not found at {chunks_path}")
                return False
            
            # Load the text chunks
            with open(chunks_path, 'rb') as f:
                self.text_chunks = pickle.load(f)
            
            # Load the FAISS index - use the embeddings we initialized in __init__
            self.vector_store = FAISS.load_local(
                folder_path=str(Path(index_path).parent),
                embeddings=self.embeddings, # This should now work
                index_name=Path(index_path).stem,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False

class ConversationContext:
    """Class to maintain conversation context for follow-up questions."""
    def __init__(self):
        self.context_stack = []
        self.max_context_length = 5
        self.last_query_results = None # New field to store last query results
    
    def add_context(self, question: str, answer: str, sql_query: str = None, query_results: List[Dict] = None):
        """Add a new Q&A pair to the context stack."""
        context = {
            "question": question,
            "answer": answer,
            "sql_query": sql_query,
            "query_results": query_results, # Store the actual data
            "timestamp": datetime.now().isoformat()
        }
        self.context_stack.append(context)
        if len(self.context_stack) > self.max_context_length:
            self.context_stack.pop(0)
        self.last_query_results = query_results # Update last results
    
    def get_relevant_context(self, new_question: str) -> str:
        """Get the most relevant context for a new question."""
        if not self.context_stack:
            return ""
        
        last_context = self.context_stack[-1]
        context_str = f"Previous Question: {last_context['question']}\nPrevious Answer: {last_context['answer']}\n"
        
        # Add relevant data from previous query if available
        if last_context.get('query_results'):
            try:
                # Extract key numerical values from previous results
                data_summary = []
                for row in last_context['query_results'][:3]: # Just first few rows
                    for k, v in row.items():
                        if isinstance(v, (int, float)):
                            data_summary.append(f"{k}: {v}")
                        elif isinstance(v, str) and v.replace('.', '', 1).isdigit():
                            data_summary.append(f"{k}: {float(v):.2f}")
                
                if data_summary:
                    context_str += "Relevant Data from Previous Query:\n" + "\n".join(data_summary) + "\n"
            except Exception as e:
                logger.error(f"Error summarizing previous query results: {e}")
        
        return context_str
    
    def clear_context(self):
        """Clear the conversation context."""
        self.context_stack = []
        self.last_query_results = None

class ChatHistoryManager:
    """Class to manage saving chat history to database."""
    
    def __init__(self):
        self.connection = None
        self.setup_connection()
    
    def setup_connection(self):
        """Establish connection to history database."""
        try:
            self.connection = mysql.connector.connect(**HISTORY_DB_CONFIG)
            if self.connection.is_connected():
                logger.info("Connected to chat history database")
                self._create_table_if_not_exists()
        except Error as e:
            logger.error(f"Error connecting to history database: {e}")
    
    def _create_table_if_not_exists(self):
        """Create the chat_history table if it doesn't exist."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS chat_history (
            id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL,
            query_id VARCHAR(36),
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sql_query TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(create_table_query)
            self.connection.commit()
            cursor.close()
        except Error as e:
            logger.error(f"Error creating chat_history table: {e}")
    
    def save_to_history(self, user_id: str, query_id: str, question: str, answer: str, sql_query: str = None):
        """Save a chat interaction to the database."""
        if not self.connection or not self.connection.is_connected():
            self.setup_connection()
        
        try:
            cursor = self.connection.cursor()
            insert_query = """
            INSERT INTO chat_history (id, user_id, query_id, question, answer, sql_query)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                str(uuid.uuid4()),
                user_id,
                query_id,
                question,
                answer, # Now saving both question and answer
                sql_query
            ))
            self.connection.commit()
            cursor.close()
            logger.info("Chat history saved to database")
        except Error as e:
            logger.error(f"Error saving chat history: {e}")
    
    def get_user_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Retrieve chat history for a specific user."""
        if not self.connection or not self.connection.is_connected():
            self.setup_connection()
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            select_query = """
            SELECT id, query_id, question, answer, sql_query, created_at
            FROM chat_history
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """
            cursor.execute(select_query, (user_id, limit))
            history = cursor.fetchall()
            cursor.close()
            
            # Format the results
            formatted_history = []
            for item in history:
                formatted_history.append({
                    "id": item["id"],
                    "query_id": item["query_id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "sql_query": item["sql_query"],
                    "timestamp": item["created_at"].isoformat() if item["created_at"] else None
                })
            
            return formatted_history
        except Error as e:
            logger.error(f"Error fetching user history: {e}")
            return []
    
    def close_connection(self):
        """Close the database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("History database connection closed")

class FollowUpHandler:
    """Handles follow-up question detection and processing."""
    
    def __init__(self, rag_system: 'PowerPartsRAGSystem'):
        self.rag_system = rag_system
        self.conversation_context = ConversationContext()
    
    def is_follow_up(self, question: str) -> bool:
        """Determine if a question is a follow-up to the previous conversation."""
        if not self.conversation_context.context_stack:
            return False
        
        question_clean = question.strip().lower()
        
        # Filter out greetings or unrelated small talk
        greeting_phrases = [
            'hello', 'hi', 'hey', 'good morning', 'good evening',
            'good afternoon', 'thank you', 'thanks', 'ok', 'okay'
        ]
        if any(greet in question_clean for greet in greeting_phrases) and len(question_clean.split()) <= 5:
            return False
        
        # Check for explicit follow-up words
        follow_up_words = [
            'then', 'also', 'what about', 'how about', 'after', 'following',
            'based on', 'from that', 'from this', 'those', 'these'
        ]
        if any(word in question_clean for word in follow_up_words):
            return True
        
        # Get the last context
        last_context = self.conversation_context.context_stack[-1]
        
        prompt = f"""
        You are an assistant that detects whether a new user query is a follow-up to a previous question.

        Only answer "yes" or "no".

        Do NOT classify greetings or general chat (like "hello", "thank you", "good evening") as follow-ups.

        Previous Question: {last_context['question']}
        Previous Answer Summary: {last_context['answer'][:300]}...
        Previous Data Summary: {self._summarize_data(last_context.get('query_results'))}

        New Question: {question}

        Only say "yes" if the new question:
        1. Clearly builds upon the data, intent, or scope of the previous interaction.
        2. Requests analysis, extension, or clarification of previous results.
        3. Is NOT a general greeting or a completely unrelated standalone question.
        """
        
        try:
            response = self.rag_system.call_llm_api(
                messages=[
                    {"role": "system", "content": "You are a follow-up question detector."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            return response.lower().strip() == 'yes'
        except Exception as e:
            logger.error(f"Error in is_follow_up: {e}")
            return False
    
    def _summarize_data(self, query_results: Optional[List[Dict]]) -> str:
        """Summarize query results for context."""
        if not query_results:
            return "No data available"
        
        summary = []
        for row in query_results[:3]: # Only first few rows
            row_summary = []
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    row_summary.append(f"{k}: {v}")
                elif isinstance(v, str) and v.replace('.', '', 1).isdigit():
                    row_summary.append(f"{k}: {float(v):.2f}")
            if row_summary:
                summary.append(", ".join(row_summary))
        
        return "\n".join(summary) if summary else "No numerical data available"
    
    def rewrite_follow_up(self, question: str) -> str:
        """Rewrite a follow-up question to include relevant context."""
        context = self.conversation_context.get_relevant_context(question)
        
        prompt = f"""
        Rewrite the follow-up question to be standalone while preserving intent.
        Return ONLY the rewritten question.

        Conversation Context:
        {context}

        Follow-up Question: {question}

        Guidelines:
        - If the question references previous numbers/data, include those values
        - Keep all business terms and technical names unchanged
        - Maintain the original intent and tone
        - Do not rewrite if it's a greeting or off-topic message — just return as-is
        """
        
        try:
            response = self.rag_system.call_llm_api(
                messages=[
                    {"role": "system", "content": "You rewrite follow-up questions to include necessary context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error in rewrite_follow_up: {e}")
            return question

class PowerPartsRAGSystem:
    def __init__(self):
        # Initialize OpenAI client first
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize vector store manager (which initializes embeddings)
        self.vector_store = VectorStoreManager()
        
        # Then setup other components
        self.db_connection = None
        self.setup_db_connection()
        self.follow_up_handler = FollowUpHandler(self)
        self.history_manager = ChatHistoryManager()
        self.user_manager = UserManager()
    
    def setup_db_connection(self):
        """Establish a connection to the MySQL database."""
        try:
            self.db_connection = mysql.connector.connect(**DB_CONFIG)
            if self.db_connection.is_connected():
                logger.info(f"Connected to MySQL database")
        except Error as e:
            logger.error(f"Error while connecting to MySQL: {e}")
            raise
    
    def close_db_connection(self):
        """Close the database connection if it exists."""
        if self.db_connection and self.db_connection.is_connected():
            self.db_connection.close()
            logger.info("MySQL connection closed")
    
    def call_llm_api(self, messages, temperature=0.7, max_tokens=1000, response_format=None):
        """
        Call LLM API, trying Ollama first and falling back to OpenAI if needed.
        """
        # Try Ollama first
        try:
            logger.info("Trying Ollama API...")
            ollama_data = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}"}
            response = requests.post(OLLAMA_API_URL, json=ollama_data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result["message"]["content"]
            else:
                logger.warning(f"Ollama API returned status {response.status_code}: {response.text}")
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Ollama API failed, falling back to OpenAI: {e}")
            
            # Fall back to OpenAI
            try:
                logger.info("Using OpenAI API as fallback...")
                openai_params = {
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                if response_format:
                    openai_params["response_format"] = response_format
                
                response = self.client.chat.completions.create(**openai_params)
                return response.choices[0].message.content
            except Exception as openai_error:
                logger.error(f"OpenAI API also failed: {openai_error}")
                raise Exception(f"Both Ollama and OpenAI APIs failed: {openai_error}")
    
    def classify_question(self, user_question: str) -> str:
        """
        Classify the user's question as either 'greeting' or 'analytical'.
        Returns 'greeting' for conversational/greeting questions, 'analytical' for data questions.
        """
        prompt = f"""
        Classify the following user question as either 'greeting' or 'analytical':

        1. 'greeting' - If the question is a greeting, introduction, small talk, or doesn't require data analysis.
        Examples:
        - "Hi there"
        - "Who are you?"
        - "What can you do?"
        - "How are you?"
        - "Tell me about yourself"
        - "Good morning"
        - "Thanks for your help"
        - "Hello"
        - "Good afternoon"
        - "What's up?"
        - "Can you help me?"
        - "What is this system about?"
        - "Explain what you do"
        - "Tell me about yourself"
        - "Tell me something about you"

        2. 'analytical' - If the question requires data analysis, database querying, or business insights.
        Examples:
        - "Show me sales trends"
        - "What's our dead inventory value?"
        - "List active AMCs in Kolkata"
        - "Compare sales between regions"
        - "What parts are in model XYZ?"
        - "Give me inventory data"
        - "Show AMC status distribution"
        - "Analyze sales performance"

        User Question: {user_question}

        Respond ONLY with either 'greeting' or 'analytical'. No other text or explanation.
        """
        
        try:
            response = self.call_llm_api(
                messages=[
                    {"role": "system", "content": "You are a question classifier that determines if a question is a greeting or requires data analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            classification = response.lower().strip()
            return classification if classification in ['greeting', 'analytical'] else 'analytical'
        
        except Exception as e:
            logger.error(f"Error in classify_question: {e}")
            return 'analytical' # Default to analytical if classification fails
    
    def generate_greeting_response(self, user_question: str) -> str:
        """Generate a professional greeting response based on the user's input."""
        prompt = f"""
        You are an AI assistant for Power Parts Private Limited. The user has asked a greeting or introductory question.
        Generate a professional, friendly response that:
        1. Acknowledges the user's input
        2. Briefly explains your capabilities
        3. Guides the user toward asking data-related questions
        4. Maintains a professional tone
        5. Is concise (3-5 sentences max)

        User's input: {user_question}

        Example responses:
        - "Hello! I'm your Power Parts data assistant. I can help analyze inventory, sales, AMCs, and more. How can I assist you today?"
        - "Good morning! I specialize in Power Parts business data analysis. I can provide insights on inventory, sales, and contracts. What would you like to explore?"
        - "Thanks for reaching out! I'm here to help with Power Parts data queries. I can analyze trends, generate reports, and answer business questions. What information do you need?"

        Respond in a similar professional but friendly style based on the user's input.
        """
        
        try:
            response = self.call_llm_api(
                messages=[
                    {"role": "system", "content": "You are a professional AI assistant for Power Parts Private Limited that handles greetings and introductory questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error in generate_greeting_response: {e}")
            return "Hello! I'm your Power Parts data assistant. How can I help you today?"
    
    def get_table_rankings(self, user_question: str) -> List[Dict]:
        """Use LLM to analyze the user's question and rank relevant tables."""
        schema_str = json.dumps(TABLE_SCHEMAS, indent=2)
        
        prompt = f"""
        Analyze the following user question and rank the tables from most relevant to least relevant based on the provided database schema.
        Return your response as a JSON array with objects containing 'table_name' and 'relevance_score' (1-10) fields.

        User Question: {user_question}

        Database Schema:
        {schema_str}

        Example Response:
        {{
        "tables": [
            {{"table_name": "sales_invoice_details", "relevance_score": 9}},
            {{"table_name": "dead_inventory_merge", "relevance_score": 5}}
        ]
        }}
        """
        
        try:
            response = self.call_llm_api(
                messages=[
                    {"role": "system", "content": "You are a database expert that analyzes questions and determines which tables are most relevant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response)
            return result.get("tables", [])
        
        except Exception as e:
            logger.error(f"Error in get_table_rankings: {e}")
            return []
    
    def clean_sql_query(self, query: str) -> str:
        """Clean the SQL query by removing markdown code blocks and other non-SQL text."""
        # Remove markdown code blocks
        query = re.sub(r'```sql|```', '', query, flags=re.IGNORECASE)
        # Remove any leading/trailing whitespace
        query = query.strip()
        # Remove semicolons if present (they can cause issues with some MySQL connectors)
        query = query.rstrip(';')
        return query
    
    def generate_sql_query(self, user_question: str, relevant_tables: List[str]) -> str:
        """Generate a SQL query based on the user's question and relevant tables."""
        # Get schema information for the relevant tables
        table_info = []
        for table in relevant_tables:
            if table in TABLE_SCHEMAS:
                table_info.append({
                    "table_name": table,
                    "description": TABLE_SCHEMAS[table]["description"],
                    "columns": TABLE_SCHEMAS[table]["columns"]
                })
        
        prompt =  f"""
You are an expert MySQL query generator for PowerParts Private Ltd's database system. Generate ONLY precise, correct SQL queries based on actual data patterns and business context.

CRITICAL DATA PATTERN ANALYSIS FROM ACTUAL DATA:

    TABLE: engine_sales_records (PRIMARY ENGINE SALES/CUSTOMER DATA)
    Real Data Patterns:
    - engine_serial_number: Alphanumeric IDs like 'DGBH6012U10635V', 'JGAF5191U22161V'
    - model_number: Technical model codes like '4008TAG2A', '2806C-E18TAG1A', '4006-23TAG2A'
    - engine_family: Specific families like '4000 series', '2000 series', '1300 series' (NOT 'Engine')
    - alternator_make: Brand names of alternator manufacturers
    - original_equipment_manufacturer: OEM names like 'CAPTIVA', 'FGW', 'PERKINS'
    - rating_in_kilowatt_amperes: Power ratings like 125, 250, 500 (numeric values in kVA)
    - captiva_invoice_date: Date when genset was invoiced/sold by Captiva
    - in_service_date: Date when genset became operational (use for sales timing analysis)
    - inside_warranty: 'NON-WARRANTY' or 'WARRANTY' (NOT 'Yes'/'No')
    - customer_name: 'COAL INDIA BHAVAN', 'Indian Dairy Products Pvt Ltd', 'AMP Urban Facility Services'
    - customer_segment: 'GOVERNMENT SECTOR', 'MANUFACTURING', 'REAL ESTATE', 'HEALTHCARE'
    - customer_city: 'Kolkata', 'Mumbai', 'Delhi' (actual city locations)
    - customer_state: 'West Bengal', 'Maharashtra', 'Delhi'
    - customer_region: 'East', 'West', 'North', 'South' (geographical regions)
    - branch: Branch office names responsible for customer
    - number_of_running_hours: Total operational hours (numeric)
    - last_routine_service_date: Date of last scheduled service
    - under_amc_need_to_be_updated: 'Yes', 'No', or blank (AMC coverage status)
    - engine_prefix_number: Numeric codes like '204', '101', '3003' (for parts compatibility)

    TABLE: sales_invoice_details (FINANCIAL TRANSACTIONS/BILLING DATA)
    Real Data Patterns:
    - warehouse: 'Kolkata', 'Gurugram' (dispatch locations)
    - financial_year: '2023-2024', '2024-2025' (YYYY-YYYY format)
    - customer_segment: Business classifications matching engine_sales_records
    - invoice_number: Structured IDs like 'PPLCU2324IN0001', 'PPLGU2425IN0002'
    - invoice_date: Standard datetime format 'YYYY-MM-DD HH:MM:SS'
    - invoice_customer_name: Customer names (may differ slightly from engine_sales_records)
    - invoice_billing_address: Complete addresses with state/pincode
    - invoice_billing_gstin_number: GST identification numbers like '21AACCS7101B1Z8'
    - hsn_code: Product classification codes like '84212300', '84799090'
    - part_number: Part identifiers like '4627133', 'PW83805', 'T427303'
    - part_name: 'OIL FILTER', 'FUEL FILTER', 'ENGINE - 1106', 'SEAL'
    - unit_price: Price per unit (decimal values)
    - quantity: Transaction quantities (integer)
    - cgst_in_percent, sgst_in_percent, igst_in_percent: Tax rates (decimal 0.09 = 9%)
    - cgst_amount, sgst_amount, igst_amount: Calculated tax amounts
    - discount_in_percent: Discount rate applied
    - discount_amount: Monetary discount value
    - taxable_amount: Pre-tax amount
    - net_amount: Final post-tax total
    - invoice_type: 'Engine and Parts Sales', 'AMC', 'Stock Transfer'
    - invoice_shipping_city: Delivery destinations

    TABLE: dead_inventory_merge (NON-MOVING INVENTORY MANAGEMENT)
    Real Data Patterns:
    - part_number: Mixed formats like '23100008', '16SE70A', '1817810C1'
    - part_name: 'NUT', 'WASHER', 'WOODRUFF KEY', 'JOINT', 'BOLT METRIC'
    - stock_quantity: Integer stock levels like 18, 19, 9, 8, 6
    - landed_price: Decimal cost values like 31.00, 31.11, 90.00, 18380.83
    - total_price: Calculated as (stock_quantity * landed_price)
    - bin_number: Storage codes like '15D', '8D', 'Not Available'
    - sub_category: Engine series like '400 SERIES', '4000 SERIES', '1300 SERIES'
    - market_value: Current market pricing (decimal values)
    - remarks: Additional notes (often location-based)
    - region_name: 'Gurugram', 'Kolkata' (inventory regions)

    TABLE: annual_maintainance_contracts_merge (AMC/SERVICE CONTRACT MANAGEMENT)
    Real Data Patterns:
    - region_name: 'Gurugram', 'Kolkata' (service regions)
    - amc_status: 'Active', 'Closed', 'Under Renewal' (contract lifecycle status)
    - customer_name: Customer organization names
    - customer_address: Complete addresses with locations and pincodes
    - customer_location: City names like 'PARADEEP', 'KOLKATA', 'GUWAHATI'
    - customer_state: 'Odisha', 'West Bengal', 'Assam', 'Jharkhand'
    - supported_by: Field engineer names like 'RANJEET GURU', 'KUNAL DEY'
    - customer_contact_person_name: Designated contact persons with titles
    - customer_contact_person_mobile_number: 10-digit mobile numbers
    - amc_or_oem: 'AMC', 'O&M' (service contract type)
    - original_equipment_manufacturer: OEM partner names
    - make_by: 'PERKINS', 'CUMMINS', 'DEUTZ' (engine manufacturers)
    - dg_set_power_rating: Power ratings like 75, 125, 250, 500, 1000 (kVA)
    - number_of_sets_sold: Quantity (typically 1)
    - engine_family: '1300', '2000', '3000', '4000' (numeric engine series)
    - engine_model_number: Specific models like '1004T', '2806C-E18 TAG1A'
    - engine_serial_number: Alphanumeric serial numbers
    - perkins_dispatched_date: Equipment dispatch dates (DD-MM-YYYY format)
    - type_of_visit: 'Quarterly', 'Monthly', 'Bi Monthly', 'BREAKDOWN VISIT'
    - amc_period_from: Contract start dates (DD-MM-YYYY format)
    - amc_expiry_date: Contract end dates (DATETIME format for comparisons)
    - order_value_including_gst: Contract values in INR (54380, 325484, 613600)
    - monthly_cost: Monthly AMC charges in INR
    - engine_prefix_number: Numeric codes for parts compatibility

    TABLE: parts_number_details (PARTS CATALOG/COMPATIBILITY DATA)
    Real Data Patterns:
    - node_name: Reference list names for specific engine builds
    - part_name: Component descriptions like 'CRANKCASE', 'CAP', 'DOWEL'
    - latest_part_number: Updated part number versions
    - part_number: Standard part identifiers
    - engine_prefix_number: Links to engine compatibility (matches other tables)

   
CRITICAL CORRECTIONS TO COMMON MISTAKES

1. Engine Family Filtering Rules
- WRONG: WHERE engine_family = 'Engine'
- CORRECT: Do NOT filter by engine_family when counting engines
- REASON: All engines have specific families like '4000 series', '2000 series'

2. Segment vs Region Confusion
- WRONG: GROUP BY region (when asked about segments)
- CORRECT: GROUP BY customer_segment
- SEGMENT VALUES: 'GOVERNMENT SECTOR', 'MANUFACTURING', 'REAL ESTATE'
- REGION VALUES: 'East', 'West', 'North', 'South'

3. Government Segment Filtering
- WRONG: WHERE customer_segment = 'Government'
- CORRECT: WHERE customer_segment = 'GOVERNMENT SECTOR'

4. Best Selling Model/Engine Queries
- WRONG: Query sales_invoice_details for engine models
- CORRECT: Use engine_sales_records table for engine model analysis
- CORRECT QUERY:
SELECT model_number, COUNT(*)
FROM engine_sales_records
GROUP BY model_number
ORDER BY COUNT(*) DESC
LIMIT 1

5. Engine Sales Count Queries
- WRONG: Use sales_invoice_details for engine counts
- CORRECT: Use engine_sales_records table directly for engine counts
- CORRECT QUERY:
SELECT COUNT(*)
FROM engine_sales_records
WHERE YEAR(in_service_date) = 2024

6. Dead Inventory Value Calculations
- WRONG: ORDER BY landed_price (for top inventory by value)
- WRONG: SUM(total_price) (incorrect column name)
- CORRECT: ORDER BY (market_value * stock_quantity) DESC (for top inventory by total value)
- CORRECT: SUM(market_value * stock_quantity) (for total inventory value)

7. AMC Renewal Rate Calculation
- WRONG: Count all records without considering unique customers
- CORRECT CUSTOMER-LEVEL CALCULATION:
SELECT ROUND(
    (COUNT(DISTINCT CASE
        WHEN amc_status IN ('Active', 'Under Renewal') THEN customer_name
    END) / COUNT(DISTINCT customer_name)) * 100,
2) AS percentage_of_customers_with_active_amcs
FROM annual_maintainance_contracts_merge;

8. Warranty Status Values
- WRONG ASSUMPTION: inside_warranty = 'Yes'/'No'
- ACTUAL VALUES: inside_warranty = 'WARRANTY' or 'NON-WARRANTY'
- CORRECT QUERY: WHERE inside_warranty = 'WARRANTY' (for warranty coverage)

9. Date Field Handling
- WRONG: Using STR_TO_DATE for amc_expiry_date
- CORRECT: amc_expiry_date is already DATETIME, use direct comparison
- CORRECT: WHERE amc_expiry_date < CURDATE() (for expired contracts)

FIXED QUERY PATTERNS FOR COMMON BUSINESS QUESTIONS

1. Total Engines Sold in Year
SELECT COUNT(*)
FROM engine_sales_records
WHERE YEAR(in_service_date) = 2024

2. Segment Distribution
SELECT customer_segment, COUNT(*)
FROM engine_sales_records
GROUP BY customer_segment

3. Government Sector Engines
SELECT COUNT(*)
FROM engine_sales_records
WHERE customer_segment = 'GOVERNMENT SECTOR'

4. Best Selling Engine Model
SELECT model_number, COUNT(*)
FROM engine_sales_records
GROUP BY model_number
ORDER BY COUNT(*) DESC
LIMIT 1

5. Top Dead Inventory by Value
SELECT part_number, part_name, (market_value * stock_quantity) as total_value
FROM dead_inventory_merge
ORDER BY total_value DESC
LIMIT 5

6. Inventory Amount by Location
SELECT SUM(market_value * stock_quantity)
FROM dead_inventory_merge
WHERE region_name = 'Kolkata'

7. AMC Renewal Rate
SELECT ROUND(
    (COUNT(CASE WHEN amc_status IN ('Active', 'Under Renewal') THEN 1 END) / COUNT(*)) * 100,
    2
) as renewal_rate_percent
FROM annual_maintainance_contracts_merge

8. Expired AMCs
SELECT customer_name
FROM annual_maintainance_contracts_merge
WHERE amc_expiry_date < CURDATE()

9. Parts Below Minimum Stock with Last Restock Date

    WHEN ASKED: Questions like:
    - "Which part numbers have inventory levels below the minimum threshold, and what are their current stock quantities and last restock dates?"
    - "List parts that are low in stock and when they were last moved or restocked."

    LOGIC TO FOLLOW:

    1. INVENTORY SOURCE:
    Use `dead_inventory_merge` table to fetch part_number, part_name, and stock_quantity.

    2. MINIMUM THRESHOLD:
    Apply a fixed threshold logic: stock_quantity < 10.
    (Unless a specific threshold column is present in schema, which is not the case here.)

    3. RESTOCK DATE LOGIC:
    Use `sales_invoice_details` table to estimate last restock based on part movement.
    Take the latest invoice_date for each part_number as the last restock date.

    4. JOIN STRATEGY:
    Use LEFT JOIN between `dead_inventory_merge` and `sales_invoice_details` on part_number.
    This ensures even parts that have never been sold are included (null restock date).

    5. GROUPING:
    Group by part_number, part_name, and stock_quantity to ensure correct aggregation.

    FIXED QUERY PATTERN:

    SELECT
        di.part_number,
        di.part_name,
        di.stock_quantity,
        MAX(sid.invoice_date) AS last_restock_date
    FROM
        dead_inventory_merge di
    LEFT JOIN
        sales_invoice_details sid ON di.part_number = sid.part_number
    WHERE
        di.stock_quantity < 10
    GROUP BY
        di.part_number, di.part_name, di.stock_quantity
    ORDER BY
        di.stock_quantity ASC;

Parts Demand Forecasting with Value Projections
    WHEN ASKED: Questions like:

    "Forecast the demand for the top 20 parts based on sales trends over the last 12 months"
    "What will be the projected sales for parts in the next 3, 6, or 12 months?"
    "Show me demand forecasting with growth rates and inventory recommendations"
    "Predict future parts demand based on historical sales data"
    "Generate sales forecasts for parts with trend analysis"

    LOGIC TO FOLLOW:

    TIME PERIOD: Use last 12 months of data from sales_invoice_details table with invoice_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH).
    DATA SOURCE: Primary table is sales_invoice_details with columns: part_number, part_name, invoice_date, quantity, net_amount, unit_price, invoice_customer_name.
    FILTERING CRITERIA: Include only records where part_number IS NOT NULL and invoice_type contains 'Parts' or 'Engine' to focus on parts sales.
    MONTHLY AGGREGATION: Group data by part_number, part_name, year, and month to create monthly sales patterns with SUM(quantity), SUM(net_amount), AVG(unit_price), and COUNT(DISTINCT customers).
    TREND ANALYSIS: Calculate growth rates by comparing recent 6 months vs previous 6 months performance for both quantity and value metrics.
    SEASONALITY FACTOR: Apply seasonality adjustments based on active months: 1.0 for ≥6 months, 0.8 for ≥3 months, 0.6 for <3 months.
    FORECAST PERIODS: Generate predictions for 3-month, 6-month, and 12-month periods using formula: (avg_monthly * period) * (1 + growth_rate) * seasonality_factor.
    DEMAND CONSISTENCY: Classify parts as 'High', 'Medium', or 'Low' consistency based on active selling months to assess reliability.
    INVENTORY RECOMMENDATIONS: Provide actionable insights like 'High Growth - Stock Up', 'Declining - Reduce Stock', 'Stable - Maintain Current Levels' based on growth patterns.
    RESULT ORDERING: Sort by historical_value_12m DESC and LIMIT to top 20 parts to focus on most valuable items for business impact.

    FIXED QUERY PATTERN:

WITH monthly_sales AS (
    SELECT
        sid.part_number,
        sid.part_name,
        YEAR(sid.invoice_date) as sales_year,
        MONTH(sid.invoice_date) as sales_month,
        SUM(sid.quantity) as monthly_quantity,
        SUM(sid.net_amount) as monthly_value,
        AVG(sid.unit_price) as avg_unit_price,
        COUNT(DISTINCT sid.invoice_customer_name) as monthly_customers
    FROM sales_invoice_details sid
    WHERE
        sid.invoice_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
        AND sid.part_number IS NOT NULL
        AND sid.part_number != ''
        AND (sid.invoice_type LIKE '%Parts%' OR sid.invoice_type LIKE '%Engine%')
    GROUP BY sid.part_number, sid.part_name, YEAR(sid.invoice_date), MONTH(sid.invoice_date)
),

parts_trend_analysis AS (
    SELECT
        part_number,
        part_name,
        -- Historical Performance (Last 12 months)
        SUM(monthly_quantity) as total_quantity_12m,
        SUM(monthly_value) as total_value_12m,
        AVG(monthly_quantity) as avg_monthly_quantity,
        AVG(monthly_value) as avg_monthly_value,
        AVG(avg_unit_price) as avg_unit_price,
        COUNT(DISTINCT CONCAT(sales_year, '-', sales_month)) as active_months,
        SUM(monthly_customers) as total_customer_interactions,
       
        -- Trend Calculations (Simple Linear Trend)
        -- Recent 6 months vs Previous 6 months growth
        SUM(CASE WHEN STR_TO_DATE(CONCAT(sales_year, '-', sales_month, '-01'), '%Y-%m-%d')
                      >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
                 THEN monthly_quantity ELSE 0 END) as recent_6m_quantity,
        SUM(CASE WHEN STR_TO_DATE(CONCAT(sales_year, '-', sales_month, '-01'), '%Y-%m-%d')
                      < DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
                 THEN monthly_quantity ELSE 0 END) as previous_6m_quantity,
                 
        SUM(CASE WHEN STR_TO_DATE(CONCAT(sales_year, '-', sales_month, '-01'), '%Y-%m-%d')
                      >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
                 THEN monthly_value ELSE 0 END) as recent_6m_value,
        SUM(CASE WHEN STR_TO_DATE(CONCAT(sales_year, '-', sales_month, '-01'), '%Y-%m-%d')
                      < DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
                 THEN monthly_value ELSE 0 END) as previous_6m_value
    FROM monthly_sales
    GROUP BY part_number, part_name
),

forecast_calculations AS (
    SELECT
        *,
        -- Growth Rate Calculations
        CASE
            WHEN previous_6m_quantity > 0
            THEN ((recent_6m_quantity - previous_6m_quantity) / previous_6m_quantity) * 100
            ELSE 0
        END as quantity_growth_rate_percent,
       
        CASE
            WHEN previous_6m_value > 0
            THEN ((recent_6m_value - previous_6m_value) / previous_6m_value) * 100
            ELSE 0
        END as value_growth_rate_percent,
       
        -- Seasonality Factor (simplified)
        CASE
            WHEN active_months >= 6 THEN 1.0
            WHEN active_months >= 3 THEN 0.8
            ELSE 0.6
        END as seasonality_factor,
       
        -- Demand Consistency Score
        CASE
            WHEN active_months >= 10 THEN 'High'
            WHEN active_months >= 6 THEN 'Medium'
            ELSE 'Low'
        END as demand_consistency
    FROM parts_trend_analysis
    WHERE total_quantity_12m > 0
)

SELECT
    part_number,
    part_name,
   
    -- Historical Performance
    total_quantity_12m as historical_quantity_12m,
    ROUND(total_value_12m, 2) as historical_value_12m,
    ROUND(avg_monthly_quantity, 2) as avg_monthly_quantity,
    ROUND(avg_monthly_value, 2) as avg_monthly_value,
    ROUND(avg_unit_price, 2) as avg_unit_price,
   
    -- Trend Analysis
    ROUND(quantity_growth_rate_percent, 2) as quantity_growth_rate_percent,
    ROUND(value_growth_rate_percent, 2) as value_growth_rate_percent,
    demand_consistency,
   
    -- 3-Month Forecast
    ROUND(
        (avg_monthly_quantity * 3) *
        (1 + (quantity_growth_rate_percent / 100)) *
        seasonality_factor, 0
    ) as forecasted_quantity_3m,
   
    ROUND(
        (avg_monthly_value * 3) *
        (1 + (value_growth_rate_percent / 100)) *
        seasonality_factor, 2
    ) as forecasted_value_3m,
   
    -- 6-Month Forecast
    ROUND(
        (avg_monthly_quantity * 6) *
        (1 + (quantity_growth_rate_percent / 100)) *
        seasonality_factor, 0
    ) as forecasted_quantity_6m,
   
    ROUND(
        (avg_monthly_value * 6) *
        (1 + (value_growth_rate_percent / 100)) *
        seasonality_factor, 2
    ) as forecasted_value_6m,
   
    -- 12-Month Forecast
    ROUND(
        (avg_monthly_quantity * 12) *
        (1 + (quantity_growth_rate_percent / 100)) *
        seasonality_factor, 0
    ) as forecasted_quantity_12m,
   
    ROUND(
        (avg_monthly_value * 12) *
        (1 + (value_growth_rate_percent / 100)) *
        seasonality_factor, 2
    ) as forecasted_value_12m,
   
    -- Investment Planning
    ROUND(
        ((avg_monthly_value * 12) *
        (1 + (value_growth_rate_percent / 100)) *
        seasonality_factor) * 0.3, 2
    ) as recommended_inventory_investment_12m,
   
    -- Risk Assessment
    CASE
        WHEN quantity_growth_rate_percent > 20 AND demand_consistency = 'High' THEN 'High Growth - Stock Up'
        WHEN quantity_growth_rate_percent > 0 AND demand_consistency = 'High' THEN 'Steady Growth - Monitor'
        WHEN quantity_growth_rate_percent < -10 THEN 'Declining - Reduce Stock'
        WHEN demand_consistency = 'Low' THEN 'Irregular Demand - Caution'
        ELSE 'Stable - Maintain Current Levels'
    END as inventory_recommendation

FROM forecast_calculations
ORDER BY historical_value_12m DESC
LIMIT 20;


TABLE SELECTION LOGIC

    1. Engine Sales/Installation Questions → engine_sales_records
    - Engine counts, model analysis, customer segments, regional distribution
    - Customer information, warranty status, service history
    - NOT sales_invoice_details (that's for financial transactions)

    2. Parts Catalog/Specifications → parts_number_details
    - Part specifications, engine component lists, compatibility
    - Use engine_prefix_number for linking to engines

    3. Inventory Analysis → dead_inventory_merge
    - Stock levels, inventory values, warehouse analysis
    - Use (market_value * stock_quantity) for total inventory value

    4. Sales Revenue/Financial → sales_invoice_details
    - Revenue analysis, financial performance, customer payments
    - Invoice details, tax calculations, billing information
    - Use for monetary calculations, not engine counts

    5. AMC/Service Contracts → annual_maintainance_contracts_merge
    - Service contracts, AMC revenue, renewal rates
    - Use amc_status for active/closed analysis
    - Contract expiry tracking, customer support management

BUSINESS INTELLIGENCE QUERIES WITH CORRECTED PATTERNS

1. Engine Performance Analysis
- Total engines sold: COUNT(*) FROM engine_sales_records
- Best selling model: GROUP BY model_number FROM engine_sales_records
- Segment distribution: GROUP BY customer_segment FROM engine_sales_records

2. Inventory Optimization
- Dead stock value: SUM(market_value * stock_quantity) FROM dead_inventory_merge
- Top inventory items: ORDER BY (market_value * stock_quantity) DESC
- Location-wise inventory: GROUP BY region_name

3. AMC Analytics
- Renewal rate: (Active contracts / Total contracts) * 100
- Revenue tracking: SUM(monthly_cost * 12) for annual revenue
- Service frequency: GROUP BY type_of_visit
- Expiry tracking: WHERE amc_expiry_date BETWEEN dates

4. Financial Analysis
- Parts vs Engine sales: GROUP BY invoice_type FROM sales_invoice_details
- Customer revenue: GROUP BY invoice_customer_name
- Regional sales: GROUP BY warehouse

AMC EXPIRY DATE HANDLING RULES

1. DO NOT use STR_TO_DATE - amc_expiry_date is already DATETIME format
2. For expired contracts: WHERE amc_expiry_date < CURDATE()
3. For expiring soon: WHERE amc_expiry_date BETWEEN CURDATE() AND (CURDATE() + INTERVAL 30 DAY)
4. For earliest expiry: ORDER BY amc_expiry_date ASC with future dates only
5. Always exclude NULL dates: WHERE amc_expiry_date IS NOT NULL

JOINING TABLES LOGIC

1. engine_sales_records ↔ annual_maintainance_contracts_merge
- Join ON engine_serial_number OR customer_name
- Use for customer service history analysis

2. engine_sales_records ↔ parts_number_details
- Join ON engine_prefix_number
- Use for parts compatibility analysis

3. sales_invoice_details ↔ dead_inventory_merge
- Join ON part_number
- Use for sales vs inventory analysis

VALIDATION RULES - PREVENT COMMON MISTAKES

1. NEVER use engine_family = 'Engine' filter for engine counts
2. Use customer_segment (not customer_region) for business segment analysis
3. Use engine_sales_records for engine-related queries
4. Use (market_value * stock_quantity) for inventory value calculations
5. Use actual segment values: 'GOVERNMENT SECTOR', 'MANUFACTURING', 'REAL ESTATE'
6. Use actual warranty values: 'WARRANTY', 'NON-WARRANTY'
7. Use amc_status for AMC analysis: 'Active', 'Closed', 'Under Renewal'
8. Select correct table based on question context
9. Handle date fields properly with YEAR() function for year extraction
10. Use proper value calculations for financial and inventory metrics
11. Include both 'Active' and 'Under Renewal' for renewal rate calculations
12. Use direct datetime comparison for amc_expiry_date (no string conversion)

STRUCTURED INSTRUCTION BLOCKS FOR SPECIFIC QUERIES

PARTS SALES VALUE (CUSTOMER-WISE)

Query Table: sales_invoice_details only
Invoice Filtering: WHERE invoice_type = 'Parts Sales'
Revenue Column: Use net_amount (final billed amount)
Financial Year Filtering: Use financial_year = 'YYYY-YYYY'
Grouping: Group by invoice_customer_name for breakdown
Purpose: Used only for parts sales, not engines or AMCs

Example: Top 10 Customers by Parts Sales (FY 2023-2024)
SELECT invoice_customer_name, SUM(net_amount) AS total_sales_value
FROM sales_invoice_details
WHERE invoice_type = 'Parts Sales'
  AND financial_year = '2023-2024'
GROUP BY invoice_customer_name
ORDER BY total_sales_value DESC
LIMIT 10;

DO NOT:
- Use gross_amount
- Use invoice_date to filter financial year
- Use this table for AMC, engine model, or maintenance queries

ENGINE MAINTENANCE (AMC DUE / EXPIRY QUERIES)

Query Table: annual_maintainance_contracts_merge only
Maintenance Timeline: Use amc_expiry_date (DATETIME type)
Active AMCs Only: WHERE amc_status = 'Active'
Date Comparison: Use CURDATE() and DATE_ADD()

Example: Engines Due for Maintenance in Next 30 Days
SELECT engine_serial_number, amc_expiry_date
FROM annual_maintainance_contracts_merge
WHERE amc_expiry_date IS NOT NULL
  AND amc_status = 'Active'
  AND amc_expiry_date BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL 30 DAY);

DO NOT:
- Use imaginary fields like next_scheduled_visit
- Filter by engine_family, customer_state unless explicitly asked
- Include Closed or NULL-status AMCs

Trigger Questions:
- "Which AMCs are expiring soon?"
- "Next maintenance due engines?"
- "Engines scheduled for service?"

ENGINE CUSTOMERS WITHOUT ACTIVE AMC

Use Tables:
- engine_sales_records → engine purchasers
- annual_maintainance_contracts_merge → AMC status info

Matching Field: customer_name
Active AMC Definition: amc_status = 'Active' (case-insensitive)

Trigger Questions:
- "Identify customers who purchased engines but do not have an active AMC agreement"
- "List customers with engine purchases but no valid AMC"
- "Who has bought engines but doesn't have active service contracts?"

ENFORCED SQL QUERY TEMPLATE:
SELECT DISTINCT esr.customer_name
FROM engine_sales_records esr
LEFT JOIN (
    SELECT DISTINCT customer_name
    FROM annual_maintainance_contracts_merge
    WHERE LOWER(amc_status) = 'active'
) amc ON esr.customer_name = amc.customer_name
WHERE esr.customer_name IS NOT NULL
  AND amc.customer_name IS NULL;

CUSTOMER AMC ANALYSIS – ACTIVE AMC BUT NO RECENT PURCHASE

Trigger Keywords:
- "customers", "not made a purchase", "no invoice", "last 12 months"
- AND includes: "active AMC" or "valid AMC"

Target Intent: Find customers who have an active AMC but have NOT made any parts or engine purchases in the last 12 months.

ENFORCED SQL QUERY:
SELECT
    amc.customer_name,
    amc.customer_address,
    amc.customer_location,
    amc.customer_state,
    amc.amc_status,
    amc.amc_expiry_date,
    amc.supported_by,
    amc.region_name
FROM annual_maintainance_contracts_merge amc
LEFT JOIN (
    SELECT DISTINCT invoice_customer_name
    FROM sales_invoice_details
    WHERE invoice_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
) recent_sales ON amc.customer_name = recent_sales.invoice_customer_name
WHERE LOWER(amc.amc_status) = 'active'
  AND recent_sales.invoice_customer_name IS NULL
  AND amc.customer_name IS NOT NULL;

FIELD MAPPING AND STATUS DEFINITIONS

Fixed Field Mappings:
- engine_sales_records.customer_name = annual_maintainance_contracts_merge.customer_name

Valid Status Definitions:
- Only AMC entries with LOWER(amc_status) = 'active' are considered active
- Ignore 'Closed', 'Under Renewal', or NULL statuses

Never Do Rules:
- Never fabricate columns like next_scheduled_visit
- Never guess AMC validity based on amc_expiry_date unless explicitly asked
- Never use partial filters or vague joins
- Never join unrelated tables like parts or service invoices

PERFORMANCE NOTES

- Use LEFT JOIN + IS NULL to find engine buyers with no active AMC
- Ensure customer_name fields are indexed in both tables for faster execution
- Use DISTINCT when dealing with duplicate customer records
- Apply appropriate date filters to improve query performance

AUTOMATIC TRIGGER LOGIC

Automatically apply specific query patterns when user asks:
- "Without active AMC" → Engine customers without AMC query
- "Engine buyers not under service contract" → Engine customers without AMC query
- "Identify customers missing AMC" → Engine customers without AMC query
- Any variant containing both "engine" and "not active" or "no AMC" → Engine customers without AMC query


    User Question: {user_question}

    Available Tables: {[table['table_name'] for table in table_info]}

    ANALYSIS STEPS:
    1. Identify the core business question and determine the PRIMARY table needed
    2. Map question keywords to correct table:
       - "engines sold/installed/count" → engine_sales_records
       - "segment distribution" → engine_sales_records.customer_segment
       - "government sector" → engine_sales_records WHERE customer_segment = 'GOVERNMENT SECTOR'
       - "best selling model" → engine_sales_records.model_number
       - "inventory value/amount" → dead_inventory_merge with (market_value * stock_quantity)
       - "AMC renewal rate" → annual_maintainance_contracts_merge.amc_status
       - "parts catalog" → parts_number_details
       - "revenue/financial" → sales_invoice_details
    3. Use exact column names and values from actual data patterns
    4. Apply correct calculations and aggregations
    5. Avoid common mistakes listed above
    6. Generate precise SQL without unnecessary filters

    Return ONLY the SQL query without any explanation:
    """

        
        try:
            response = self.call_llm_api(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a MySQL expert who generates precise queries based on actual data patterns. Focus on selecting the correct table and using exact column values. Avoid common mistakes like filtering by non-existent values or using wrong tables for specific business questions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.01, # Very low for maximum accuracy
                max_tokens=1500
            )
            
            # Clean the SQL query before returning
            return self.clean_sql_query(response)
        
        except Exception as e:
            logger.error(f"Error in generate_sql_query: {e}")
            return ""
    
    def execute_sql_query(self, query: str) -> Tuple[List[Dict], List[str]]:
        """Execute the SQL query and return the results."""
        if not query:
            return [], ["Empty query provided"]
        
        if not self.db_connection or not self.db_connection.is_connected():
            self.setup_db_connection()
        
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Convert Decimal objects to float for JSON serialization
            converted_results = []
            for row in results:
                converted_row = {}
                for key, value in row.items():
                    if isinstance(value, Decimal):
                        converted_row[key] = float(value)
                    else:
                        converted_row[key] = value
                converted_results.append(converted_row)
            
            column_names = [col[0] for col in cursor.description] if cursor.description else []
            cursor.close()
            
            return converted_results, column_names
        
        except Error as e:
            logger.error(f"Error executing SQL query: {e}")
            return [], [str(e)]
        except Exception as e:
            logger.error(f"Unexpected error executing SQL query: {e}")
            return [], [str(e)]
    
    def generate_final_answer(self, user_question: str, query: str, query_results: List[Dict], column_names: List[str]) -> str:
        """Generate a professional answer based on the query results with chunking for large datasets."""
        if not query_results:
            return "No results found for your query."
        
        try:
            # Estimate token count (approximate: 1 token ≈ 4 characters)
            results_json = json.dumps({
                "query_results": query_results,
                "column_names": column_names
            }, cls=CustomJSONEncoder)
            
            estimated_tokens = len(results_json) // 4
            
            # If results are too large, use chunking and summarization
            if estimated_tokens > 20000: # Conservative threshold to avoid rate limits
                logger.info(f"Large dataset detected ({estimated_tokens} estimated tokens), using chunked summarization")
                return self._generate_chunked_answer(user_question, query, query_results, column_names)
            
            # For smaller datasets, use the original approach
            return self._generate_direct_answer(user_question, query, query_results, column_names)
        
        except Exception as e:
            logger.error(f"Error in generate_final_answer: {e}")
            return f"An error occurred while generating the answer: {str(e)}"
    
    def _generate_direct_answer(self, user_question: str, query: str, query_results: List[Dict], column_names: List[str]) -> str:
        """Original implementation for smaller datasets."""
        results_str = json.dumps({
            "query_results": query_results,
            "column_names": column_names
        }, indent=2, cls=CustomJSONEncoder)
        
        prompt = f"""
        You are a professional Chartered Accountant and financial analyst for PowerParts Private Ltd.
        Based on the following user question, SQL query, and query results, generate a concise professional summary that will be perfectly formatted for the frontend UI.

        CRITICAL FORMATTING RULES:
        Always use this exact bullet point format: "▸ " followed by a space (e.g., ▸ Total Active Customers: 397)
        Do not use markdown symbols like *, **, -, or #
        Format all numbers with commas (e.g., 1,000 not 1000)
        Use clear section headings that are relevant to the actual query results
        Put each data point on its own line
        Ensure all currency values are represented in Indian Rupees (₹)
        Keep language simple, clear, and professional
        Be concise but insightful

        User Question: {user_question}

        SQL Query Used:
        {query}

        Query Results:
        {results_str}

        Column Names: {column_names}

        Generate your response using ONLY the actual query results provided above.
        """
        
        try:
            response = self.call_llm_api(
                messages=[
                    {
                        "role": "system",
                        "content": """You are a professional data analyst that explains query results in clear,
                        business-appropriate language formatted for a specific frontend UI."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            answer = response
            return self._clean_response_formatting(answer)
        
        except Exception as e:
            logger.error(f"Error in _generate_direct_answer: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_chunked_answer(self, user_question: str, query: str, query_results: List[Dict], column_names: List[str]) -> str:
        """Generate answer for large datasets using chunking and summarization."""
        logger.info(f"Starting chunked summarization for {len(query_results)} rows")
        
        # Split data into manageable chunks (adjust chunk_size based on your data complexity)
        chunk_size = 300 # Smaller chunks for safety
        chunks = [query_results[i:i + chunk_size] for i in range(0, len(query_results), chunk_size)]
        
        logger.info(f"Split into {len(chunks)} chunks of {chunk_size} rows each")
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_summary = self._summarize_chunk(user_question, query, chunk, column_names, i+1, len(chunks))
            chunk_summaries.append(chunk_summary)
            
            # Add small delay to avoid rate limits
            time.sleep(0.5)
        
        # Generate final summary from chunk summaries
        final_summary = self._generate_final_summary(user_question, query, chunk_summaries, len(query_results))
        
        return final_summary
    
    def _summarize_chunk(self, user_question: str, query: str, chunk_data: List[Dict], column_names: List[str],
                        chunk_number: int, total_chunks: int) -> str:
        """Summarize a single chunk of data."""
        chunk_str = json.dumps({
            "query_results": chunk_data,
            "column_names": column_names
        }, indent=2, cls=CustomJSONEncoder)
        
        prompt = f"""
        You are summarizing a portion of a large dataset for PowerParts Private Ltd.
        This is chunk {chunk_number} of {total_chunks}. Focus on key patterns, trends, and important numbers.

        User Question: {user_question}
        SQL Query: {query}

        Data Chunk {chunk_number}/{total_chunks}:
        {chunk_str}

        Provide a concise summary of this data chunk focusing on:
        1. Key numerical values and their significance
        2. Notable patterns or trends
        3. Any outliers or exceptional values
        4. Business implications specific to this chunk

        Keep your summary focused and factual. Return ONLY the summary text.
        """
        
        try:
            response = self.call_llm_api(
                messages=[
                    {"role": "system", "content": "You summarize data chunks concisely and accurately."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500 # Shorter responses for chunks
            )
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"Error summarizing chunk {chunk_number}: {e}")
            return f"Summary unavailable for chunk {chunk_number} due to error."
    
    def _generate_final_summary(self, user_question: str, query: str, chunk_summaries: List[str], total_rows: int) -> str:
        """Generate final summary from chunk summaries."""
        all_summaries = "\n\n".join([f"Chunk {i+1} Summary:\n{summary}" for i, summary in enumerate(chunk_summaries)])
        
        prompt = f"""
        You are a professional Chartered Accountant and financial analyst for PowerParts Private Ltd.
        Based on the user question and summaries of all data chunks, generate a comprehensive final analysis.

        User Question: {user_question}
        SQL Query: {query}
        Total Rows Analyzed: {total_rows:,}

        Chunk Summaries:
        {all_summaries}

        Generate a professional summary that:
        1. Synthesizes insights from all chunk summaries
        2. Provides overall trends and patterns
        3. Highlights key business implications
        4. Uses proper formatting with "▸ " bullet points
        5. Includes recommendations based on the complete dataset

        Follow the same formatting rules as the direct analysis method.
        """

        try:
            response = self.call_llm_api(
                messages=[
                    {"role": "system", "content": "You synthesize chunk summaries into comprehensive analyses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )

            answer = response
            return self._clean_response_formatting(answer)

        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            # Fallback: return concatenated chunk summaries
            return f"Comprehensive analysis of {total_rows:,} rows:\n\n" + "\n\n".join(chunk_summaries)

    def _clean_response_formatting(self, answer: str) -> str:
        """Clean and standardize the response formatting."""
        # Remove all markdown formatting
        answer = answer.replace("**", "").replace("*", "").replace("###", "").replace("##", "").replace("#", "")

        # Fix bullet points - ensure they use the correct format
        lines = answer.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                # Convert dash bullets to our format
                cleaned_lines.append(f"▸ {line[2:]}")
            elif line.startswith('• '):
                # Convert bullet bullets to our format
                cleaned_lines.append(f"▸ {line[2:]}")
            elif line.startswith('▸') and not line.startswith('▸ '):
                # Ensure proper spacing after our bullet
                cleaned_lines.append(f"▸ {line[1:].strip()}")
            elif line.startswith(' ▸'):
                # Handle indented bullets and fix spacing
                cleaned_lines.append(f" ▸ {line[5:].strip()}")
            else:
                cleaned_lines.append(line)

        # Rejoin lines
        answer = '\n'.join(cleaned_lines)

        # Ensure proper spacing between sections
        sections = answer.split('\n\n')
        cleaned_sections = []

        for section in sections:
            section = section.strip()
            if section: # Only add non-empty sections
                cleaned_sections.append(section)

        answer = '\n\n'.join(cleaned_sections)

        # Remove any references to example data that might have leaked through
        example_phrases = [
            "engine sales performance over time",
            "June 2024",
            "August 2023",
            "March 2025",
            "₹1,58,60,76,44.80",
            "₹37,82,60,19.02",
            "₹10,17,37,403.41"
        ]

        answer_lower = answer.lower()
        for phrase in example_phrases:
            if phrase.lower() in answer_lower:
                # If example data is found, this indicates the AI used example instead of real data
                logger.warning("Example data detected in response, this should not happen")

        return answer

    def determine_visualization_type(self, columns: List[str], data: List[Dict]) -> Optional[str]:
        """Determine the best visualization type based on the data."""
        if not data:
            return None

        # Count numeric columns
        numeric_cols = 0
        for col in columns:
            # Check if all values in this column are numeric
            all_numeric = True
            for row in data:
                val = row.get(col)
                if val is None:
                    continue
                if not isinstance(val, (int, float)) and not (isinstance(val, str) and val.replace('.', '', 1).isdigit()):
                    all_numeric = False
                    break

            if all_numeric:
                numeric_cols += 1

        if numeric_cols == 0:
            return None
        elif numeric_cols == 1:
            return "bar" # Simple bar chart for single numeric column
        elif numeric_cols >= 2:
            # If we have a date/time column, use line chart
            for col in columns:
                for row in data:
                    val = row.get(col)
                    if isinstance(val, str) and any(c in val for c in ['-', '/']):
                        return "line"
            return "bar" # Default to bar chart for multiple numeric columns

        return None

    def process_question(self, user_question: str, is_follow_up: bool = False, context: Optional[Dict] = None, user_id: Optional[str] = None) -> Dict:
        """Full RAG pipeline to process a user question with optional follow-up context."""
        process_log = []

        try:
            # Generate a query ID for this interaction
            query_id = str(uuid.uuid4())

            # Step 0: Handle follow-up questions
            final_question = user_question
            if is_follow_up or self.follow_up_handler.is_follow_up(user_question):
                is_follow_up = True
                process_log.append(f"Follow-up question detected: {user_question}")

                final_question = self.follow_up_handler.rewrite_follow_up(user_question)
                process_log.append(f"Original question: {user_question}")
                process_log.append(f"Rewritten question: {final_question}")


            # Step 1: Classify the question
            question_type = self.classify_question(final_question)
            process_log.append(f"Question classified as: {question_type}")

            if question_type == "greeting":
                process_log.append("Generating greeting response...")
                greeting_response = self.generate_greeting_response(final_question)

                # Add to context even if greeting for continuity
                self.follow_up_handler.conversation_context.add_context(
                    question=user_question,
                    answer=greeting_response
                )

                # Save greeting to history if user is authenticated
                if user_id:
                    self.history_manager.save_to_history(
                        user_id=user_id,
                        query_id=query_id,
                        question=user_question,
                        answer=greeting_response
                    )

                return {
                    "status": "success",
                    "question_type": "greeting",
                    "answer": greeting_response,
                    "process_log": process_log,
                    "is_follow_up": is_follow_up,
                    "query_id": query_id
                }

            # Step 2: Rank relevant tables
            process_log.append("Analyzing question to identify relevant data tables...")
            table_rankings = self.get_table_rankings(final_question)
            relevant_tables = [t["table_name"] for t in table_rankings if t["relevance_score"] >= 5]

            if not relevant_tables:
                return {
                    "status": "error",
                    "message": "No relevant tables found for the question.",
                    "process_log": process_log,
                    "is_follow_up": is_follow_up,
                    "query_id": query_id
                }

            process_log.append(f"Identified relevant tables: {', '.join(relevant_tables)}")

            # Step 3: Generate SQL query
            process_log.append("Generating SQL query for the question...")
            query = self.generate_sql_query(final_question, relevant_tables)
            if not query:
                return {
                    "status": "error",
                    "message": "Could not generate a valid SQL query for the question.",
                    "process_log": process_log,
                    "is_follow_up": is_follow_up,
                    "query_id": query_id
                }

            process_log.append(f"Generated SQL query: {query[:100]}...") # Log first 100 chars

            # Step 4: Execute query
            process_log.append("Executing query against the database...")
            query_results, column_names = self.execute_sql_query(query)

            if isinstance(query_results, str): # Error case
                return {
                    "status": "error",
                    "message": f"Error executing query: {query_results}",
                    "process_log": process_log,
                    "sql_query": query,
                    "is_follow_up": is_follow_up,
                    "query_id": query_id
                }

            process_log.append(f"Query executed successfully, returned {len(query_results)} rows")

            # Step 5: Generate final answer
            process_log.append("Generating natural language response...")
            final_answer = self.generate_final_answer(final_question, query, query_results, column_names)

            self.follow_up_handler.conversation_context.add_context(
                question=user_question, # Store original question
                answer=final_answer,
                sql_query=query,
                query_results=query_results # Include the actual data
            )

            # Determine visualization type
            visualization_type = self.determine_visualization_type(column_names, query_results)
            if visualization_type:
                process_log.append(f"Identified {visualization_type} chart as appropriate visualization")

            # Add to conversation context
            self.follow_up_handler.conversation_context.add_context(
                question=user_question, # Store original question
                answer=final_answer,
                sql_query=query
            )

            # Save to history database
            if user_id:
                self.history_manager.save_to_history(
                    user_id=user_id,
                    query_id=query_id,
                    question=user_question,
                    answer=final_answer,
                    sql_query=query
                )

            return {
                "status": "success",
                "question_type": "analytical",
                "table_rankings": table_rankings,
                "sql_query": query,
                "columns": column_names,
                "data": query_results,
                "answer": final_answer,
                "visualization_type": visualization_type,
                "process_log": process_log,
                "is_follow_up": is_follow_up,
                "original_question": user_question if is_follow_up else None,
                "query_id": query_id
            }

        except Exception as e:
            logger.error(f"Error in process_question: {e}")
            process_log.append(f"Error occurred: {str(e)}")
            return {
                "status": "error",
                "message": f"An unexpected error occurred: {str(e)}",
                "process_log": process_log,
                "is_follow_up": is_follow_up,
                "query_id": query_id if 'query_id' in locals() else str(uuid.uuid4())
            }

    # Initialize the RAG system
rag_system = PowerPartsRAGSystem()

@app.on_event("shutdown")
def shutdown_event():
    rag_system.close_db_connection()
    rag_system.history_manager.close_connection()
    rag_system.user_manager.close_connection()

    # Authentication Endpoints
@app.post("/api/auth/register")
async def register_user(request: Request):
    """Endpoint to register a new user."""
    try:
        data = await request.json()
        email = data.get("email", "").strip()
        name = data.get("name", "").strip()
        password = data.get("password", "").strip()

        if not email or not password or not name:
            raise HTTPException(status_code=400, detail="Email, name and password are required")

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            raise HTTPException(status_code=400, detail="Invalid email format")

        if len(password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

        success = rag_system.user_manager.register_user(email, name, password)
        if not success:
            raise HTTPException(status_code=400, detail="Registration failed (email may already exist)")

        return JSONResponse(content={"status": "success", "message": "User registered successfully"})

    except Exception as e:
        logger.error(f"Error in user registration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/login")
async def login_user(request: Request):
    """Endpoint to authenticate a user."""
    try:
        data = await request.json()
        email = data.get("email", "").strip()
        password = data.get("password", "").strip()

        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password are required")

        user = rag_system.user_manager.authenticate_user(email, password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return JSONResponse(content={
            "status": "success",
            "user": {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"]
            }
        })

    except Exception as e:
        logger.error(f"Error in user login: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# API Endpoints
@app.post("/api/query")
async def submit_query(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        is_follow_up = data.get("is_follow_up", False)
        context = data.get("context", None)
        user_id = request.headers.get("X-User-ID")

        if not question:
            raise HTTPException(status_code=400, detail="Question is required")

        query_id = str(uuid.uuid4())
        logger.info(f"Processing query {query_id} for user {user_id}")

        result = rag_system.process_question(
            user_question=question,
            is_follow_up=is_follow_up,
            context=context,
            user_id=user_id
        )

        if result["status"] != "success":
            raise HTTPException(status_code=400, detail=result.get("message", "Query failed"))

        response_data = {
            "status": "completed",
            "query_id": query_id,
            "question_type": result.get("question_type", "analytical"),
            "answer": result.get("answer", "No answer generated"),
            "sql_query": result.get("sql_query", ""),
            "columns": result.get("columns", []),
            "data": result.get("data", []),
            "visualization_type": result.get("visualization_type"),
            "process_log": result.get("process_log", []),
            "original_question": result.get("original_question", question),
            "is_follow_up": is_follow_up,
            "timestamp": datetime.now()
            }

            # 🔁 Also store in query_cache
        with query_lock:
            query_cache[query_id] = {
                "status": "completed",
                "result": response_data,
                "timestamp": time.time()
            }

        logger.info(f"Query {query_id} completed")
        return JSONResponse(content=jsonable_encoder(response_data))

    except Exception as e:
        logger.error(f"Error in submit_query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/clear-context")
async def clear_conversation_context():
    """Endpoint to clear the conversation context."""
    try:
        rag_system.follow_up_handler.conversation_context.clear_context()
        return JSONResponse(content={"status": "success", "message": "Conversation context cleared"})
    except Exception as e:
        logger.error(f"Error clearing context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/query/{query_id}")
async def get_query_result(query_id: str):
    """Endpoint to check the status of a query and get results when ready."""
    try:
        with query_lock:
            result = query_cache.get(query_id)

        if not result:
            raise HTTPException(status_code=404, detail="Query not found")

        if result["status"] == "completed":
            response_data = {
                "status": "completed",
                "question_type": result["result"].get("question_type", "analytical"),
                "answer": result["result"].get("answer", "No answer generated"),
                "sql_query": result["result"].get("sql_query", ""),
                "columns": result["result"].get("columns", []),
                "data": result["result"].get("data", []),
                "visualization_type": result["result"].get("visualization_type"),
                "process_log": result["result"].get("process_log", []),
                # Add these new fields for history
                "original_question": result["result"].get("original_question", ""),
                "is_follow_up": result["result"].get("is_follow_up", False),
                "timestamp": datetime.now().isoformat()
            }
            return JSONResponse(content=jsonable_encoder(response_data))

        elif result["status"] == "failed":
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failed",
                    "error": result.get("error", result["result"].get("message", "Unknown error")),
                    "process_log": result.get("process_log", [])
                }
            )

        else: # processing
            return JSONResponse(content={
                "status": "processing",
                "process_log": result.get("process_log", [])
            })

    except Exception as e:
        logger.error(f"Error in get_query_result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat-history")
async def get_chat_history(request: Request):
    """Endpoint to fetch user-specific chat history."""
    try:
        # Get user ID from authorization header or token
        user_id = request.headers.get("X-User-ID")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID required")

        history = rag_system.history_manager.get_user_history(user_id)
        return JSONResponse(content=jsonable_encoder(history))

    except Error as e:
        logger.error(f"Error fetching chat history: {e}")
        raise HTTPException(status_code=500, detail="Error fetching chat history")
    except Exception as e:
        logger.error(f"Unexpected error fetching chat history: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

# Frontend Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main frontend interface."""
    return templates.TemplateResponse("index.html", {"request": request})

# Error handlers
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
        status_code=404,
        content={"message": exc.detail}
        )

@app.exception_handler(500)
async def server_error_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"Server error: {str(exc)}")
    JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
