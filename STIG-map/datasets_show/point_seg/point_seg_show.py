from PySide2.QtWidgets import QApplication, QMainWindow, QTableView, QVBoxLayout, QWidget, QPushButton
from PySide2.QtCore import Qt, QAbstractTableModel, QModelIndex

class TableModel(QAbstractTableModel):
    def __init__(self, data=None, headers=None):
        super().__init__()
        self._data = data or []  # Start with an empty list if no data is passed
        self.headers = headers or []

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)  # Ensure column count matches the number of headers

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if role == Qt.DisplayRole:
            return self._data[row][col] if row < len(self._data) else ""
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[section]  # Return the header names for columns
            else:
                return f"{section + 1}"
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if index.isValid() and role == Qt.EditRole:
            row, col = index.row(), index.column()
            if row < len(self._data) and col < len(self.headers):
                # Update the data at the specific row and column
                self._data[row][col] = value
                # Emit signal to inform view of data change
                self.dataChanged.emit(index, index, [Qt.DisplayRole])
                return True
        return False

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        # Enable item to be selectable, enabled, and editable
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def add_row(self, row_data):
        """
        Add a new row to the model
        """
        row_position = self.rowCount()
        self.beginInsertRows(QModelIndex(), row_position, row_position)  # Notify view of new row
        self._data.append(row_data)  # Add the row data to the model
        self.endInsertRows()  # End insert operation, notify view

    def clear_data(self):
        """清空数据并刷新表格视图"""
        self.beginResetModel()
        self._data = []
        self.endResetModel()