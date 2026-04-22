from flask import Flask, request, jsonify
import uuid
import datetime
from typing import List, Dict, Optional

app = Flask(__name__)

# In-memory storage for leave records
leave_records = []

# Helper function to calculate days between dates
def calculate_days(start_date: str, end_date: str) -> int:
    try:
        start = datetime.date.fromisoformat(start_date)
        end = datetime.date.fromisoformat(end_date)
        return (end - start).days + 1
    except ValueError:
        return 0

# Helper function to validate date format
def is_valid_date(date_str: str) -> bool:
    try:
        datetime.date.fromisoformat(date_str)
        return True
    except ValueError:
        return False

# Helper function to validate leave type
def is_valid_leave_type(leave_type: str) -> bool:
    valid_types = [
        'annual_leave', 'sick_leave', 'personal_leave', 
        'marriage_leave', 'maternity_leave', 'paternity_leave', 'bereavement_leave'
    ]
    return leave_type in valid_types

# Helper function to validate status
def is_valid_status(status: str) -> bool:
    valid_statuses = ['pending', 'approved', 'rejected']
    return status in valid_statuses

# Helper function to find leave by ID
def find_leave_by_id(leave_id: str) -> Optional[Dict]:
    for record in leave_records:
        if record['id'] == leave_id:
            return record
    return None

# Helper function to get current ISO datetime string
def get_current_datetime() -> str:
    return datetime.datetime.now().isoformat() + 'Z'

@app.route('/api/leaves', methods=['GET'])
def get_leaves():
    try:
        # Get query parameters
        employee_name = request.args.get('employee_name')
        leave_type = request.args.get('leave_type')
        status = request.args.get('status')
        
        # Filter records based on query parameters
        filtered_records = leave_records.copy()
        
        if employee_name:
            filtered_records = [r for r in filtered_records if r['employee_name'] == employee_name]
        
        if leave_type:
            if not is_valid_leave_type(leave_type):
                return jsonify({'detail': 'Invalid leave_type parameter'}), 400
            filtered_records = [r for r in filtered_records if r['leave_type'] == leave_type]
        
        if status:
            if not is_valid_status(status):
                return jsonify({'detail': 'Invalid status parameter'}), 400
            filtered_records = [r for r in filtered_records if r['status'] == status]
        
        return jsonify(filtered_records), 200
        
    except Exception as e:
        return jsonify({'detail': 'Server error'}), 500


@app.route('/api/leaves', methods=['POST'])
def create_leave():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['employee_id', 'employee_name', 'leave_type', 'start_date', 'end_date', 'reason']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'detail': f'Missing required field: {field}'}), 400
        
        # Validate date format
        if not is_valid_date(data['start_date']):
            return jsonify({'detail': 'Invalid start_date format. Use YYYY-MM-DD'}), 400
        
        if not is_valid_date(data['end_date']):
            return jsonify({'detail': 'Invalid end_date format. Use YYYY-MM-DD'}), 400
        
        # Validate date range
        if data['end_date'] < data['start_date']:
            return jsonify({'detail': 'end_date must be greater than or equal to start_date'}), 400
        
        # Validate leave_type
        if not is_valid_leave_type(data['leave_type']):
            return jsonify({'detail': 'Invalid leave_type'}), 400
        
        # Validate reason length
        if len(data['reason']) > 200:
            return jsonify({'detail': 'Reason must be less than or equal to 200 characters'}), 422
        
        # Calculate days
        days = calculate_days(data['start_date'], data['end_date'])
        
        # Create new leave record
        new_leave = {
            'id': str(uuid.uuid4()),
            'employee_id': data['employee_id'],
            'employee_name': data['employee_name'],
            'leave_type': data['leave_type'],
            'start_date': data['start_date'],
            'end_date': data['end_date'],
            'days': days,
            'apply_time': get_current_datetime(),
            'status': 'pending',
            'approver': '',
            'reason': data['reason']
        }
        
        leave_records.append(new_leave)
        
        return jsonify(new_leave), 201
        
    except Exception as e:
        return jsonify({'detail': 'Server error'}), 500


@app.route('/api/leaves/<string:leave_id>', methods=['GET'])
def get_leave(leave_id):
    try:
        leave = find_leave_by_id(leave_id)
        if not leave:
            return jsonify({'detail': 'Leave record not found'}), 404
        
        return jsonify(leave), 200
        
    except Exception as e:
        return jsonify({'detail': 'Server error'}), 500


@app.route('/api/leaves/<string:leave_id>', methods=['PUT'])
def update_leave(leave_id):
    try:
        leave = find_leave_by_id(leave_id)
        if not leave:
            return jsonify({'detail': 'Leave record not found'}), 404
        
        # Only pending records can be updated
        if leave['status'] != 'pending':
            return jsonify({'detail': 'Cannot update record with status other than "pending"'}), 403
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['employee_id', 'employee_name', 'leave_type', 'start_date', 'end_date', 'reason']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'detail': f'Missing required field: {field}'}), 400
        
        # Validate date format
        if not is_valid_date(data['start_date']):
            return jsonify({'detail': 'Invalid start_date format. Use YYYY-MM-DD'}), 400
        
        if not is_valid_date(data['end_date']):
            return jsonify({'detail': 'Invalid end_date format. Use YYYY-MM-DD'}), 400
        
        # Validate date range
        if data['end_date'] < data['start_date']:
            return jsonify({'detail': 'end_date must be greater than or equal to start_date'}), 400
        
        # Validate leave_type
        if not is_valid_leave_type(data['leave_type']):
            return jsonify({'detail': 'Invalid leave_type'}), 400
        
        # Validate reason length
        if len(data['reason']) > 200:
            return jsonify({'detail': 'Reason must be less than or equal to 200 characters'}), 422
        
        # Update the record
        leave['employee_id'] = data['employee_id']
        leave['employee_name'] = data['employee_name']
        leave['leave_type'] = data['leave_type']
        leave['start_date'] = data['start_date']
        leave['end_date'] = data['end_date']
        leave['days'] = calculate_days(data['start_date'], data['end_date'])
        leave['reason'] = data['reason']
        
        return jsonify(leave), 200
        
    except Exception as e:
        return jsonify({'detail': 'Server error'}), 500


@app.route('/api/leaves/<string:leave_id>', methods=['DELETE'])
def delete_leave(leave_id):
    try:
        leave = find_leave_by_id(leave_id)
        if not leave:
            return jsonify({'detail': 'Leave record not found'}), 404
        
        # Only pending records can be deleted
        if leave['status'] != 'pending':
            return jsonify({'detail': 'Cannot delete record with status other than "pending"'}), 403
        
        leave_records.remove(leave)
        
        return '', 204
        
    except Exception as e:
        return jsonify({'detail': 'Server error'}), 500


@app.route('/api/leaves/<string:leave_id>/status', methods=['PATCH'])
def update_leave_status(leave_id):
    try:
        leave = find_leave_by_id(leave_id)
        if not leave:
            return jsonify({'detail': 'Leave record not found'}), 404
        
        # Only pending records can have their status updated
        if leave['status'] != 'pending':
            return jsonify({'detail': 'Cannot update status of record that is not "pending"'}), 403
        
        data = request.get_json()
        
        # Validate required fields
        if 'status' not in data:
            return jsonify({'detail': 'Missing required field: status'}), 400
        
        if 'approver' not in data:
            return jsonify({'detail': 'Missing required field: approver'}), 400
        
        # Validate status value
        if data['status'] not in ['approved', 'rejected']:
            return jsonify({'detail': 'Invalid status value. Must be "approved" or "rejected"'}), 400
        
        # Update the status and approver
        leave['status'] = data['status']
        leave['approver'] = data['approver']
        
        return jsonify(leave), 200
        
    except Exception as e:
        return jsonify({'detail': 'Server error'}), 500


if __name__ == '__main__':
    # Add some sample data for testing
    sample_leave = {
        'id': '1',
        'employee_id': 'EMP001',
        'employee_name': 'Zhang San',
        'leave_type': 'annual_leave',
        'start_date': '2023-01-01',
        'end_date': '2023-01-05',
        'days': 5,
        'apply_time': '2023-01-01T08:00:00.000Z',
        'status': 'pending',
        'approver': '',
        'reason': 'Vacation'
    }
    leave_records.append(sample_leave)
    
    app.run(debug=True, host='0.0.0.0', port=8000)
